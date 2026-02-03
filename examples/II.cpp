#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Geometry.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float                  = double;

constexpr Float X_MIN        = 0.0;
constexpr Float X_MAX        = 5.0;
constexpr Float Y_MIN        = 0.0;
constexpr Float Y_MAX        = 1.0;

constexpr Index NY           = 128;
constexpr Index NX           = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST       = 1;

constexpr Float T_END        = 3.0;
constexpr Float DT_MAX       = 1e-2;
constexpr Float CFL_MAX      = 0.5;
constexpr Float DT_WRITE     = T_END / 100.0;

constexpr Float VISC         = 1e-3;
constexpr Float RHO          = 1.0;

constexpr Circle wall        = {.x = 1.0, .y = 0.5, .r = 0.15};
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(wall.contains({.x = x, .y = y}));
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, Float /*t*/) -> Float {
  constexpr Float height = Y_MAX - Y_MIN;
  constexpr Float U      = 1.5;
  return (4.0 * U * y * (height - y)) / Igor::sqr(height);
}
constexpr Float U_AVG = quadrature([](Float y) { return U_in(y, 0.0); }, Y_MIN, Y_MAX);
constexpr Float Re    = RHO * U_AVG * (Y_MAX - Y_MIN) / VISC;

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0.0;
  outflow = 0.0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag(-NGHOST, j) * fs.curr.U(-NGHOST, j);
    outflow += fs.curr.rho_u_stag(NX + NGHOST, j) * fs.curr.U(NX + NGHOST, j);
  });
  mass_error = outflow - inflow;
}

// -------------------------------------------------------------------------------------------------
constexpr Index NMARKER = 100;
constexpr Float KAPPA   = 1.0;
constexpr Float ETA     = 1.0;
struct InterfaceMarker {
  // Initial position ^= X
  Field1D<Float, NMARKER> x0;
  Field1D<Float, NMARKER> y0;

  // Current position ^= Chi(X, t)
  Field1D<Float, NMARKER> x;
  Field1D<Float, NMARKER> y;

  // Current velocity ^= U(X, t)
  Field1D<Float, NMARKER> u;
  Field1D<Float, NMARKER> v;

  // Previous position ^= Chi(X, t-dt)
  Field1D<Float, NMARKER> x_old;
  Field1D<Float, NMARKER> y_old;

  // Response force
  Field1D<Float, NMARKER> FU;
  Field1D<Float, NMARKER> FV;

  // Jump conditins
  Field1D<Float, NMARKER> p_jump;
  Field1D<Float, NMARKER> mu_dudx_jump;
  Field1D<Float, NMARKER> mu_dvdx_jump;
  Field1D<Float, NMARKER> mu_dudy_jump;
  Field1D<Float, NMARKER> mu_dvdy_jump;
};

constexpr void calc_response_force(InterfaceMarker& marker) {
  for_each_i(marker.x, [&](Index idx) {
    marker.FU(idx) = KAPPA * (marker.x0(idx) - marker.x(idx)) - ETA * marker.u(idx);
    marker.FV(idx) = KAPPA * (marker.y0(idx) - marker.y(idx)) - ETA * marker.v(idx);
  });
}

constexpr void calc_jumps(InterfaceMarker& marker) {
  // TODO: Assue J^-1 = 1, is this a good assumption?
  constexpr Float J_inv = 1.0;

  for_each_i(marker.x, [&](Index idx) {
    const Index idx_prev = ((idx - 1) + NMARKER) % NMARKER;
    const Index idx_next = (idx + 1) % NMARKER;
    const Vector2 t1     = {
            .x = marker.x(idx) - marker.x(idx_prev),
            .y = marker.y(idx) - marker.y(idx_prev),
    };
    const auto n1    = normalize(Vector2{
           .x = -t1.y,
           .y = t1.x,
    });
    const Vector2 t2 = {
        .x = marker.x(idx_next) - marker.x(idx),
        .y = marker.y(idx_next) - marker.y(idx),
    };
    const auto n2   = normalize(Vector2{
          .x = -t2.y,
          .y = t2.x,
    });
    const Vector2 n = {
        .x = (n1.x + n2.x) / 2.0,
        .y = (n1.y + n2.y) / 2.0,
    };

    marker.p_jump(idx) = J_inv * (marker.FU(idx) * n.x + marker.FV(idx) * n.y);

    // TODO: This needs to be revised
    marker.mu_dudx_jump(idx) =
        J_inv * ((1.0 - n.x * n.x) * marker.FU(idx) + (1.0 - n.x * n.y) * marker.FV(idx)) * n.x;
    marker.mu_dvdx_jump(idx) =
        J_inv * ((1.0 - n.x * n.y) * marker.FU(idx) + (1.0 - n.y * n.y) * marker.FV(idx)) * n.x;

    marker.mu_dudy_jump(idx) =
        J_inv * ((1.0 - n.x * n.x) * marker.FU(idx) + (1.0 - n.x * n.y) * marker.FV(idx)) * n.y;
    marker.mu_dvdy_jump(idx) =
        J_inv * ((1.0 - n.x * n.y) * marker.FU(idx) + (1.0 - n.y * n.y) * marker.FV(idx)) * n.y;
  });
}

[[nodiscard]] auto write_interface(const std::string& directory, const InterfaceMarker& marker)
    -> bool {
  static Index write_counter = 0;

  if (!to_npy(Igor::detail::format("{}/marker_x_{:06}.npy", directory, write_counter), marker.x)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/marker_y_{:06}.npy", directory, write_counter), marker.y)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/marker_u_{:06}.npy", directory, write_counter), marker.u)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/marker_v_{:06}.npy", directory, write_counter), marker.v)) {
    return false;
  }

  write_counter += 1;

  return true;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  Igor::Info("Re = {:.6e}", Re);

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR           = get_output_directory();
  const auto INTERFACE_OUTPUT_DIR = OUTPUT_DIR + "interface";
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }
  if (!init_output_directory(INTERFACE_OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);
  ps.set_pressure_operator(fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  Field2D<Float, NX, NY, NGHOST> ib{};

  // Observation variables
  Float t  = 0.0;
  Float dt = DT_MAX;

  // Float mass    = 0.0;
  // Float mom_x   = 0.0;
  // Float mom_y   = 0.0;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0;
  Float p_max   = 0.0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("Immersed-wall", &ib);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");

  monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  // monitor.add_variable(&mass, "mass");
  // monitor.add_variable(&mom_x, "momentum (x)");
  // monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for_each_a<Exec::Parallel>(ib, [&](Index i, Index j) {
    ib(i, j) =
        quadrature(immersed_wall, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });

  InterfaceMarker marker{};
  for_each<0, NMARKER>([&](Index idx) {
    const Float angle = 2.0 * std::numbers::pi_v<Float> * static_cast<Float>(idx) / NMARKER;
    marker.x0(idx)    = wall.r * std::cos(angle) + wall.x;
    marker.y0(idx)    = wall.r * std::sin(angle) + wall.y;
    marker.x(idx)     = marker.x0(idx);
    marker.y(idx)     = marker.y0(idx);
    marker.u(idx) = bilinear_interpolate(fs.x, fs.ym, fs.curr.U, marker.x0(idx), marker.y0(idx));
    marker.v(idx) = bilinear_interpolate(fs.xm, fs.y, fs.curr.V, marker.x0(idx), marker.y0(idx));
  });
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  fill(fs.curr.U, 0.0);
  fill(fs.curr.V, 0.0);
  apply_velocity_bconds(fs, bconds, t);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  p_max   = max(fs.p);
  if (!data_writer.write(t)) { return 1; }
  if (!write_interface(INTERFACE_OUTPUT_DIR, marker)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    copy(marker.x, marker.x_old);
    copy(marker.y, marker.y_old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      calc_response_force(marker);
      calc_jumps(marker);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });
      apply_velocity_bconds(fs, bconds, t);
      // = Update flow field =======================================================================

      // = Correct the outflow =====================================================================
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });
      // = Correct the outflow =====================================================================

      // = Apply pressure correction ===============================================================
      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_p_iter = 0;
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      // Correct pressure
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      // Correct velocity
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        const auto dpdx  = (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx;
        const auto rho   = fs.curr.rho_u_stag(i, j);
        fs.curr.U(i, j) -= dpdx * dt / rho;
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        const auto dpdy  = (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy;
        const auto rho   = fs.curr.rho_v_stag(i, j);
        fs.curr.V(i, j) -= dpdy * dt / rho;
      });
      // = Apply pressure correction ===============================================================

      // = Update marker position ==================================================================
      for_each_i(marker.u, [&](Index idx) {
        marker.x(idx) = (marker.x(idx) + marker.x_old(idx)) / 2.0;
        marker.y(idx) = (marker.y(idx) + marker.y_old(idx)) / 2.0;

        marker.u(idx) = bilinear_interpolate(fs.x, fs.ym, fs.curr.U, marker.x(idx), marker.y(idx));
        marker.v(idx) = bilinear_interpolate(fs.xm, fs.y, fs.curr.V, marker.x(idx), marker.y(idx));

        marker.x(idx) = marker.x_old(idx) + dt * marker.u(idx);
        marker.y(idx) = marker.y_old(idx) + dt * marker.v(idx);
      });
      // = Update marker position ==================================================================
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    p_max   = max(fs.p);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
      if (!write_interface(INTERFACE_OUTPUT_DIR, marker)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
