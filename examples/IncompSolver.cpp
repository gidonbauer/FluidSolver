#include <cstddef>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 2.2;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 0.41;

constexpr Index NY              = 64;
constexpr Index NX              = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST          = 1;

constexpr Float T_END           = 8.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 5e-2;

constexpr Float U_0             = 0.0;
constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 1.0;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left =
        Dirichlet<Float>{
            .U = [](Float y, Float t) -> Float {
              IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
              constexpr auto DELTA_Y = Y_MAX - Y_MIN;
              const auto U           = 1.5 * std::sin(std::numbers::pi * t / 8.0);
              return (4.0 * U * y * (DELTA_Y - y)) / Igor::sqr(DELTA_Y);
            },
            .V = 0.0,
        },
    .right  = Neumann{},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};

// // Couette flow
// constexpr FlowBConds<Float> bconds{
//     //        LEFT            RIGHT           BOTTOM            TOP
//     .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
//     .U     = {0.0, 0.0, 0.0, U_BCOND},
//     .V     = {0.0, 0.0, 0.0, 0.0},
// };
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
  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};

  Matrix<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Matrix<Float, NX, NY, NGHOST> delta_p{};

  Float t             = 0.0;
  Float dt            = DT_MAX;

  Float U_max         = 0.0;
  Float V_max         = 0.0;
  Float div_max       = 0.0;

  Float inflow        = 0.0;
  Float outflow       = 0.0;
  Float mass_error    = 0.0;

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
  monitor.add_variable(&inflow, "inflow");
  monitor.add_variable(&outflow, "outflow");
  monitor.add_variable(&mass_error, "mass error");
  // = Output ======================================================================================

  // = Initialize pressure solver ==================================================================
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::NONE);

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = U_0; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds, t);
  calc_inflow_outflow(fs, inflow, outflow, mass_error);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
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

      // Correct the outflow
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_pressure_iter = 0;
      ps.solve(fs, div, dt, delta_p, &pressure_res, &local_pressure_iter);
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
    calc_inflow_outflow(fs, inflow, outflow, mass_error);

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finish successfully.");
}
