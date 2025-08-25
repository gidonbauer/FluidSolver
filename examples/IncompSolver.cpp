#include <cstddef>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
// #include "XDMFWriter.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 64 * 10;
constexpr Index NY              = 64;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 10.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 1.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 1e-1;

constexpr Float U_BCOND         = 1.0;
constexpr Float U_0             = 0.0;
constexpr Float VISC            = 1e-3;  // 1e-1;
constexpr Float RHO             = 0.9;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_BCOND, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

// // Couette flow
// constexpr FlowBConds<Float> bconds{
//     //        LEFT            RIGHT           BOTTOM            TOP
//     .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
//     .U     = {0.0, 0.0, 0.0, U_BCOND},
//     .V     = {0.0, 0.0, 0.0, 0.0},
// };

constexpr auto OUTPUT_DIR = "output/IncompSolver/";
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
  // XDMFWriter<Float, NX, NY, NGHOST> data_writer(
  //     Igor::detail::format("{}/solution.xdmf2", OUTPUT_DIR),
  //     Igor::detail::format("{}/solution.h5", OUTPUT_DIR),
  //     &fs.x,
  //     &fs.y);
  VTKWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  calc_rho_and_visc(fs);

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
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::RIGHT);

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U,
                             [U = fs.curr.U.view()](Index i, Index j) mutable { U(i, j) = U_0; });
  for_each_i<Exec::Parallel>(fs.curr.V,
                             [V = fs.curr.V.view()](Index i, Index j) mutable { V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);
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
  Igor::ProgressBar<Float> pbar(T_END, 67);
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
      for_each_i<Exec::Parallel>(fs.curr.U,
                                 [&,
                                  U       = fs.curr.U.view(),
                                  U_old   = fs.old.U.view(),
                                  rho     = fs.curr.rho_u_stag.view(),
                                  rho_old = fs.old.rho_u_stag.view(),
                                  drhoUdt = drhoUdt.view()](Index i, Index j) mutable {
                                   U(i, j) = (rho_old(i, j) * U_old(i, j) + dt * drhoUdt(i, j)) /
                                             rho(i, j);
                                 });
      for_each_i<Exec::Parallel>(fs.curr.V,
                                 [&,
                                  V       = fs.curr.V.view(),
                                  V_old   = fs.old.V.view(),
                                  rho     = fs.curr.rho_v_stag.view(),
                                  rho_old = fs.old.rho_v_stag.view(),
                                  drhoVdt = drhoVdt.view()](Index i, Index j) mutable {
                                   V(i, j) = (rho_old(i, j) * V_old(i, j) + dt * drhoVdt(i, j)) /
                                             rho(i, j);
                                 });
      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      // Correct the outflow
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(
          fs.ym, [&, U = fs.curr.U.view(), rho = fs.curr.rho_u_stag.view()](Index j) mutable {
            U(NX + NGHOST, j) -=
                mass_error / (rho(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
          });

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_pressure_iter = 0;
      ps.solve(fs, div, dt, delta_p, &pressure_res, &local_pressure_iter);
      pressure_iter += local_pressure_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for_each_a<Exec::Parallel>(
          fs.p, [p = fs.p.view(), delta_p = delta_p.view()](Index i, Index j) mutable {
            p(i, j) += delta_p(i, j);
          });

      for_each_i<Exec::Parallel>(
          fs.curr.U,
          [&, U = fs.curr.U.view(), delta_p = delta_p.view(), rho = fs.curr.rho_u_stag.view()](
              Index i, Index j) mutable {
            U(i, j) -= (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx * dt / rho(i, j);
          });
      for_each_i<Exec::Parallel>(
          fs.curr.V,
          [&, V = fs.curr.V.view(), delta_p = delta_p.view(), rho = fs.curr.rho_v_stag.view()](
              Index i, Index j) mutable {
            V(i, j) -= (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy * dt / rho(i, j);
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
    pbar.update(dt);
  }
  std::cout << '\n';

  Igor::Info("Solver finish successfully.");
}
