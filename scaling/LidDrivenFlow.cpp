#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 1.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Index NGHOST          = 1;

constexpr Float T_END           = 25.0;
constexpr Float DT_WRITE        = T_END / 100;
constexpr Float DT_MAX          = std::min(1e-1, DT_WRITE);
constexpr Float CFL_MAX         = 0.9;

constexpr Float U_TOP           = 1.0;
constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 1.0;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .right  = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = U_TOP, .V = 0.0},
};

// constexpr Float Re = U_TOP * (X_MAX - X_MIN) * RHO / VISC;
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
template <Index N>
auto run_simulation() -> bool {
  constexpr Index NX = 1 << N;
  constexpr Index NY = 1 << N;

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory("scaling/output") + std::to_string(N) + "/";
  if (!init_output_directory(OUTPUT_DIR)) { return false; }
  // Igor::Info("Re = {:.6e}", Re);

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  Float t             = 0.0;
  Float dt            = DT_MAX;

  Float U_max         = 0.0;
  Float V_max         = 0.0;
  Float div_max       = 0.0;

  Float pressure_res  = 0.0;
  Index pressure_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  calc_rho(fs);
  calc_visc(fs);

  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");
  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&pressure_res, "res(p)");
  monitor.add_variable(&pressure_iter, "iter(p)");
  // = Output ======================================================================================

  // = Initialize pressure solver ==================================================================
  // PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);
  ps.set_pressure_operator(fs);

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds, t);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  if (!data_writer.write(t)) { return false; }
  monitor.write();
  // = Initialize flow field =======================================================================

  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_state(fs.curr, fs.old);

    pressure_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

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
      // Boundary conditions
      apply_velocity_bconds(fs, bconds, t);

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_pressure_iter = 0;
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &pressure_res, &local_pressure_iter);
      pressure_iter += local_pressure_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) -= (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx * dt / RHO;
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) -= (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy * dt / RHO;
      });
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return false; }
    }
    monitor.write();
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  run_simulation<4>();
  run_simulation<5>();
  run_simulation<6>();
  run_simulation<7>();
  run_simulation<8>();
  run_simulation<9>();
  run_simulation<10>();
}
