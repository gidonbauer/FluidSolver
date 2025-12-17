#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 128;
constexpr Index NY              = 128;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = -1.0;
constexpr Float X_MAX           = 1.0;
constexpr Float Y_MIN           = -1.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 5.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-2;

constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 1.0;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr Float CX              = 0.0;
constexpr Float CY              = 0.0;
constexpr Float R0              = 0.1;
// constexpr auto drdt(Float t) { return 0.05 * t * t; }
// constexpr auto r(Float t) { return R0 + 1.0 / 3.0 * 0.05 * t * t * t; }
constexpr auto immersed_wall(Float x, Float y, Float r) -> Float {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(r));
}

// #define CHANNEL_FLOW
#ifdef CHANNEL_FLOW
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = 0.01, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
#else
constexpr FlowBConds<Float> bconds{
    .left   = Neumann{.clipped = true},
    .right  = Neumann{.clipped = true},
    .bottom = Neumann{.clipped = true},
    .top    = Neumann{.clipped = true},
};
#endif
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
struct IB {
  Field2D<Float, NX, NY, NGHOST> wall;
  Field2D<Float, NX + 1, NY, NGHOST> wall_u_stag;
  Field2D<Float, NX, NY + 1, NGHOST> wall_v_stag;

  Field2D<Float, NX + 1, NY, NGHOST> fU;
  Field2D<Float, NX, NY + 1, NGHOST> fV;
};

// -------------------------------------------------------------------------------------------------
auto ib_kernel(Float ib_value) -> Float {
  return 1.0 - ib_value;
  // return static_cast<Float>(ib_value < 1e-1);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_divergence_ib(const Field2D<Float, NX, NY, NGHOST>& wall,
                        const Field2D<Float, NX + 1, NY, NGHOST>& U,
                        const Field2D<Float, NX, NY + 1, NGHOST>& V,
                        Float dx,
                        Float dy,
                        Field2D<Float, NX, NY, NGHOST>& div) {
  for_each_a<Exec::Parallel>(div, [&](Index i, Index j) {
    div(i, j) =
        ((U(i + 1, j) - U(i, j)) / dx + (V(i, j + 1) - V(i, j)) / dy) * ib_kernel(wall(i, j));
  });
}

// -------------------------------------------------------------------------------------------------
void calc_conserved_quantities_ib(const FS<Float, NX, NY, NGHOST>& fs,
                                  const IB& ib,
                                  Float& mass,
                                  Float& momentum_x,
                                  Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for_each<0, NX, 0, NY>([&](Index i, Index j) {
    mass += (ib_kernel(ib.wall_u_stag(i, j)) * fs.curr.rho_u_stag(i, j) +
             ib_kernel(ib.wall_u_stag(i + 1, j)) * fs.curr.rho_u_stag(i + 1, j) +
             ib_kernel(ib.wall_v_stag(i, j)) * fs.curr.rho_v_stag(i, j) +
             ib_kernel(ib.wall_v_stag(i, j + 1)) * fs.curr.rho_v_stag(i, j + 1)) /
            4.0 * fs.dx * fs.dy;

    momentum_x +=
        (ib_kernel(ib.wall_u_stag(i, j)) * fs.curr.rho_u_stag(i, j) * fs.curr.U(i, j) +
         ib_kernel(ib.wall_u_stag(i + 1, j)) * fs.curr.rho_u_stag(i + 1, j) * fs.curr.U(i + 1, j)) /
        2.0 * fs.dx * fs.dy;
    momentum_y +=
        (ib_kernel(ib.wall_v_stag(i, j)) * fs.curr.rho_v_stag(i, j) * fs.curr.V(i, j) +
         ib_kernel(ib.wall_v_stag(i, j + 1)) * fs.curr.rho_v_stag(i, j + 1) * fs.curr.V(i, j + 1)) /
        2.0 * fs.dx * fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
void calc_ib(const FS<Float, NX, NY, NGHOST>& fs, Float r, IB& ib) {
  for_each_a<Exec::Parallel>(ib.wall_u_stag, [&](Index i, Index j) {
    ib.wall_u_stag(i, j) = quadrature([&](Float x, Float y) { return immersed_wall(x, y, r); },
                                      fs.x(i) - fs.dx / 2.0,
                                      fs.x(i) + fs.dx / 2.0,
                                      fs.y(j),
                                      fs.y(j + 1)) /
                           (fs.dx * fs.dy);
  });
  for_each_a<Exec::Parallel>(ib.wall_v_stag, [&](Index i, Index j) {
    ib.wall_v_stag(i, j) = quadrature([&](Float x, Float y) { return immersed_wall(x, y, r); },
                                      fs.x(i),
                                      fs.x(i + 1),
                                      fs.y(j) - fs.dy / 2.0,
                                      fs.y(j) + fs.dy / 2.0) /
                           (fs.dx * fs.dy);
  });
  for_each_a<Exec::Parallel>(ib.wall, [&](Index i, Index j) {
    ib.wall(i, j) = quadrature([&](Float x, Float y) { return immersed_wall(x, y, r); },
                               fs.x(i),
                               fs.x(i + 1),
                               fs.y(j),
                               fs.y(j + 1)) /
                    (fs.dx * fs.dy);
  });
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  IB ib{};
  Field2D<Float, NX, NY, NGHOST> fUi{};
  Field2D<Float, NX, NY, NGHOST> fVi{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float r       = R0;
  Float drdt    = 0.0;

  Float mass    = 0.0;
  Float mom_x   = 0.0;
  Float mom_y   = 0.0;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("wall", &ib.wall);
  data_writer.add_vector("fIB", &fUi, &fVi);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");

  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&r, "r");
  monitor.add_variable(&drdt, "dr/dt");

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  calc_ib(fs, r, ib);
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  for_each_a<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_a<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  interpolate_U(ib.fU, fUi);
  interpolate_V(ib.fV, fVi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max = abs_max(fs.curr.U);
  V_max = abs_max(fs.curr.V);
  calc_divergence_ib(ib.wall, fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  div_max = abs_max(div);
  calc_conserved_quantities_ib(fs, ib, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // = Update IB field ===========================================================================
    drdt = 0.05 * t;  // * t;
    calc_ib(fs, r, ib);
    // = Update IB field ===========================================================================

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);

      // = Calculate IB forcing term ===============================================================
      for_each_i<Exec::Parallel>(ib.fU, [&](Index i, Index j) {
        const auto x     = fs.x(i);
        const auto y     = fs.ym(j);
        const auto d     = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));

        const Float u_ib = d > 1e-8 ? drdt * (x - CX) / d : 0.0;
        ib.fU(i, j) = (fs.old.rho_u_stag(i, j) * (u_ib - fs.old.U(i, j)) / dt - drhoUdt(i, j)) *
                      ib.wall_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(ib.fV, [&](Index i, Index j) {
        const auto x     = fs.xm(i);
        const auto y     = fs.y(j);
        const auto d     = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));

        const Float v_ib = d > 1e-8 ? drdt * (y - CY) / d : 0.0;

        ib.fV(i, j) = (fs.old.rho_v_stag(i, j) * (v_ib - fs.old.V(i, j)) / dt - drhoVdt(i, j)) *
                      ib.wall_v_stag(i, j);
      });
      // = Calculate IB forcing term ===============================================================

      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });

      // = Apply IB forcing term ===================================================================
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) += ib.fU(i, j) * dt / fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) += ib.fV(i, j) * dt / fs.curr.rho_v_stag(i, j);
      });
      // = Apply IB forcing term ===================================================================

      apply_velocity_bconds(fs, bconds);

      // = Correct the outflow =====================================================================
#ifdef CHANNEL_FLOW
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });
#endif  // CHANNEL_FLOW
      // = Correct the outflow =====================================================================

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      // = Adjust div for bubble growth ============================================================
      // See: Sepahi, F., Verzicco, R., Lohse, D., Krug, D., 2024. Mass transport at gas-evolving
      // electrodes. Journal of Fluid Mechanics 983, A19. https://doi.org/10.1017/jfm.2024.51
      for_each_i<Exec::Parallel>(
          div, [&](Index i, Index j) { div(i, j) -= ib.wall(i, j) * 3.0 / r * drdt; });

      // = Apply pressure correction ===============================================================
      Index local_p_iter = 0;
      ps.set_pressure_operator(fs);
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      // Correct pressure
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      // Correct velocity
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        const auto dpdx = (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx;
        const auto rho  = fs.curr.rho_u_stag(i, j);
        // fs.curr.U(i, j) -= (dpdx * dt / rho) * (1.0 - ib.wall_u_stag(i, j));
        fs.curr.U(i, j) -= (dpdx * dt / rho);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        const auto dpdy = (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy;
        const auto rho  = fs.curr.rho_v_stag(i, j);
        // fs.curr.V(i, j) -= (dpdy * dt / rho) * (1.0 - ib.wall_v_stag(i, j));
        fs.curr.V(i, j) -= (dpdy * dt / rho);
      });
      // = Apply pressure correction ===============================================================
    }

    r += drdt * dt;
    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    interpolate_U(ib.fU, fUi);
    interpolate_V(ib.fV, fVi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max = abs_max(fs.curr.U);
    V_max = abs_max(fs.curr.V);
    calc_divergence_ib(ib.wall, fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    div_max = abs_max(div);
    calc_conserved_quantities_ib(fs, ib, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
