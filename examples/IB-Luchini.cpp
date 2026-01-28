#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "IB.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float              = double;

constexpr Float X_MIN    = 0.0;
constexpr Float X_MAX    = 5.0;
constexpr Float Y_MIN    = 0.0;
constexpr Float Y_MAX    = 1.0;

constexpr Index NY       = 128;
constexpr Index NX       = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST   = 1;

constexpr Float T_END    = 5.0;
constexpr Float DT_MAX   = 1e-2;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = T_END / 100.0;

constexpr Float VISC     = 1e-3;
constexpr Float RHO      = 1.0;

// #define IB_CHANNEL
#ifdef IB_CHANNEL
constexpr Rect wall1 = {
    .x = X_MIN,
    .y = Y_MIN,
    .w = X_MAX - X_MIN,
    .h = (Y_MAX - Y_MIN) / 4.0,
};
constexpr Rect wall2 = {
    .x = X_MIN,
    .y = Y_MIN + 3.0 * (Y_MAX - Y_MIN) / 4.0,
    .w = X_MAX - X_MIN,
    .h = (Y_MAX - Y_MIN) / 4.0,
};
#else
constexpr Circle wall1 = {.x = 1.0, .y = 0.5, .r = 0.15};
constexpr Rect wall2   = {.x = 2.75, .y = 0.25, .w = 0.5, .h = 0.5};
#endif  // IB_CHANNEL

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, [[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
#ifdef IB_CHANNEL
  if (y < (Y_MAX - Y_MIN) / 4.0 || y > 3.0 * (Y_MAX - Y_MIN) / 4.0) { return 0.0; }
  constexpr Float height = (Y_MAX - Y_MIN) / 2.0;
  constexpr Float U      = 1.5;
  const Float y_off      = y - (Y_MAX - Y_MIN) / 4.0;
  return (4.0 * U * y_off * (height - y_off)) / Igor::sqr(height);
#else
  constexpr Float height = Y_MAX - Y_MIN;
  constexpr Float U      = 1.5;
  return (4.0 * U * y * (height - y)) / Igor::sqr(height);
#endif  // IB_CHANNEL
}

constexpr Float U_AVG =
    quadrature<64>([](Float y) { return U_in(y, 0.0); }, Y_MIN, Y_MAX) / (Y_MAX - Y_MIN);
constexpr Float Re = RHO * U_AVG * (Y_MAX - Y_MIN) / VISC;

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
auto main() -> int {
  Igor::Info("Re = {:.6e}", Re);

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
  ps.set_pressure_operator(fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  // Immersed-wall
  Field2D<Float, NX, NY, NGHOST> ib{};
  Field2D<Float, NX + 1, NY, NGHOST> ib_corr_u_stag{};
  Field2D<Float, NX, NY + 1, NGHOST> ib_corr_v_stag{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

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
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for_each_a<Exec::Parallel>(ib, [&](Index i, Index j) {
    ib(i, j) = quadrature(
                   [](Float x, Float y) -> Float {
                     Point p{.x = x, .y = y};
                     return static_cast<Float>(wall1.contains(p) || wall2.contains(p));
                   },
                   fs.x(i),
                   fs.x(i + 1),
                   fs.y(j),
                   fs.y(j + 1)) /
               (fs.dx * fs.dy);
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
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      update_velocity(drhoUdt, drhoVdt, dt, fs);

      fill(ib_corr_u_stag, 0.0);
      fill(ib_corr_v_stag, 0.0);
      calc_ib_correction_shape(wall1, fs.dx, fs.dy, fs.x, fs.ym, ib_corr_u_stag);
      calc_ib_correction_shape(wall1, fs.dx, fs.dy, fs.xm, fs.y, ib_corr_v_stag);

      calc_ib_correction_shape(wall2, fs.dx, fs.dy, fs.x, fs.ym, ib_corr_u_stag);
      calc_ib_correction_shape(wall2, fs.dx, fs.dy, fs.xm, fs.y, ib_corr_v_stag);

      correct_velocity_ib_implicit(ib_corr_u_stag, ib_corr_v_stag, dt, fs);

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
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
