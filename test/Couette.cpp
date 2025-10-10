#include <cstddef>
#include <filesystem>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Utility.hpp"
#include "VTKWriter.hpp"

#include "Common.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 210;
constexpr Index NY              = 21;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 10.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 10.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 0.1;

constexpr Float U_TOP           = 1.0;
constexpr Float U_INIT          = 0.0;
constexpr Float VISC            = 1e-1;
constexpr Float RHO             = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

// Couette flow
constexpr FlowBConds<Float> bconds{
    //        LEFT            RIGHT           BOTTOM            TOP
    .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {0.0, 0.0, 0.0, U_TOP},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "test/output/Couette/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0;
  outflow = 0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag(0, j) * fs.curr.U(0, j);
    outflow += fs.curr.rho_u_stag(NX, j) * fs.curr.U(NX, j);
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
  calc_rho(fs);
  calc_visc(fs);

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};

  Matrix<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Matrix<Float, NX, NY, NGHOST> delta_p{};

  Float t          = 0.0;
  Float dt         = DT_MAX;

  Float U_max      = 0.0;
  Float V_max      = 0.0;
  Float div_max    = 0.0;

  Float p_res      = 0.0;
  Index p_iter     = 0;

  Float inflow     = 0;
  Float outflow    = 0;
  Float mass_error = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");
  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");
  monitor.add_variable(&inflow, "inflow");
  monitor.add_variable(&outflow, "outflow");
  monitor.add_variable(&mass_error, "mass error");

  VTKWriter<Float, NX, NY, NGHOST> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::RIGHT);

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = U_INIT; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Couette");
  bool solver_failed   = false;
  bool any_test_failed = false;
  while (t < T_END && !solver_failed) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_state(fs.curr, fs.old);

    p_iter = 0;
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
      apply_velocity_bconds(fs, bconds);

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_p_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        solver_failed = true;
      }
      p_iter += local_p_iter;

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

    {
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      if (std::abs(mass_error) > 1e-8) {
        if (!any_test_failed) {
          Igor::Warn("Outflow is not equal to inflow at t={:.6e}: inflow={:.6e}, outflow={:.6e}, "
                     "error={:.6e}",
                     t,
                     inflow,
                     outflow,
                     std::abs(outflow - inflow));
        }
        any_test_failed = true;
      }
    }

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  if (solver_failed) {
    Igor::Warn("Couette failed.");
    return 1;
  }

  // = Perform tests ===============================================================================
  {
    auto u_analytical = [&](Float y) -> Float {
      // NOTE: Adjustment due to the ghost cells, the dirichlet boundary condition is now enforced
      //       in the ghost cell
      const auto m = U_TOP / (1.0 + fs.dy);
      const auto n = m * fs.dy / 2.0;
      return m * y + n;
      // return U_TOP * y;
    };
    Vector<Float, NY + 2 * NGHOST> diff{};

    constexpr Index N_CHECKS                       = 3;
    constexpr std::array<size_t, N_CHECKS> i_check = {NX / 4, NX / 2, 3 * NX / 4};
    std::array<Float, N_CHECKS> L1_errors{};

    size_t counter = 0;
    for (size_t i : i_check) {
      for (Index j = -NGHOST; j < fs.curr.U.extent(1) + NGHOST; ++j) {
        diff(j + NGHOST) = std::abs(fs.curr.U(static_cast<Index>(i), j) - u_analytical(fs.ym(j)));
      }
      L1_errors[counter++] = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
    }

    constexpr Float TOL = 1e-4;
    counter             = 0;
    for (size_t i : i_check) {
      const auto err = L1_errors[counter++];
      if (err > TOL) {
        Igor::Warn("U-velocity profile at x={} does not align with analytical solution: L1-error "
                   "is {:.6e}",
                   fs.x(static_cast<Index>(i)),
                   err);
        any_test_failed = true;
      }
    }
  }

  return any_test_failed ? 1 : 0;
}
