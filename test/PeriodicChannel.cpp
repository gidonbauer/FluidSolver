#include <cstddef>
#include <type_traits>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "VTKWriter.hpp"

#include "Common.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 320;
constexpr Index NY              = 32;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 10.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 60.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 1.0;

constexpr Float U_IN            = 1.0;
constexpr Float U_INIT          = 0.0;
constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 0.5;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM           TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::PERIODIC, BCond::PERIODIC},
    .U     = {U_IN, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "test/output/PeriodicChannel/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0;
  outflow = 0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag[-NGHOST, j] * fs.curr.U[-NGHOST, j];
    outflow += fs.curr.rho_u_stag[NX + NGHOST, j] * fs.curr.U[NX + NGHOST, j];
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
  calc_rho_and_visc(fs);

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};

  Matrix<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Matrix<Float, NX, NY, NGHOST> delta_p{};

  Float t          = 0.0;
  Float dt         = DT_MAX;

  Float mass       = 0.0;
  Float mom_x      = 0.0;
  Float mom_y      = 0.0;

  Float U_max      = 0.0;
  Float V_max      = 0.0;
  Float div_max    = 0.0;

  Float p_res      = 0.0;
  Index p_iter     = 0;
  Float p_max      = 0.0;

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
  monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");
  // monitor.add_variable(&inflow, "inflow");
  // monitor.add_variable(&outflow, "outflow");
  monitor.add_variable(&mass_error, "mass error");
  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");

  VTKWriter<Float, NX, NY, NGHOST> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  // = Initialize pressure solver ==================================================================
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::RIGHT);
  // = Initialize pressure solver ==================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U[i, j] = U_INIT; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V[i, j] = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  p_max   = abs_max(fs.p);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("PeriodicChannel");
  bool failed          = false;
  bool any_test_failed = false;
  while (t < T_END && !failed) {
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
        if (std::isnan(drhoUdt[i, j])) { Igor::Panic("NaN value in drhoUdt[{}, {}]", i, j); }
        fs.curr.U[i, j] = (fs.old.rho_u_stag[i, j] * fs.old.U[i, j] + dt * drhoUdt[i, j]) /
                          fs.curr.rho_u_stag[i, j];
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        if (std::isnan(drhoVdt[i, j])) { Igor::Panic("NaN value in drhoVdt[{}, {}]", i, j); }
        fs.curr.V[i, j] = (fs.old.rho_v_stag[i, j] * fs.old.V[i, j] + dt * drhoVdt[i, j]) /
                          fs.curr.rho_v_stag[i, j];
      });

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U[NX + NGHOST, j] -=
            mass_error / (fs.curr.rho_u_stag[NX + NGHOST, j] * static_cast<Float>(NY + 2 * NGHOST));
      });

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      Index local_p_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
      }
      p_iter += local_p_iter;

      {
        if (std::any_of(delta_p.get_data(), delta_p.get_data() + delta_p.size(), [](Float x) {
              return std::isnan(x);
            })) {
          Igor::Warn("Encountered NaN value in pressure correction.");
          return 1;
        }
      }

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p[i, j] += delta_p[i, j]; });

      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U[i, j] -=
            (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx * dt / fs.curr.rho_u_stag[i, j];
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V[i, j] -=
            (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy * dt / fs.curr.rho_v_stag[i, j];
      });
    }
    t += dt;

    {
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      if (std::abs(mass_error) > 1e-8) {
        Igor::Warn("Outflow is not equal to inflow at t={:.6e}: inflow={:.6e}, outflow={:.6e}, "
                   "error={:.6e}",
                   t,
                   inflow,
                   outflow,
                   std::abs(outflow - inflow));
        any_test_failed = true;
      }
    }

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    p_max   = abs_max(fs.p);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  if (failed) {
    Igor::Warn("LaminarChannel failed.");
    return 1;
  }

  // = Perform tests ===============================================================================

  // Check U velocity field
  for_each_a(fs.curr.U, [&](Index i, Index j) {
    if (std::abs(fs.curr.U[i, j] - U_IN) > 1e-8) {
      Igor::Warn("Incorrect U-velocity at ({:.6e}, {:.6e}), expected {:.6e} but got {:.6e}",
                 fs.x[i],
                 fs.ym[j],
                 U_IN,
                 fs.curr.U[i, j]);
      any_test_failed = true;
    }
  });

  // Check V velocity field
  for_each_a(fs.curr.V, [&](Index i, Index j) {
    if (std::abs(fs.curr.V[i, j]) > 1e-8) {
      Igor::Warn("Incorrect V-velocity at ({:.6e}, {:.6e}), expected {:.6e} but got {:.6e}",
                 fs.xm[i],
                 fs.y[j],
                 0.0,
                 fs.curr.V[i, j]);
      any_test_failed = true;
    }
  });

  return any_test_failed ? 1 : 0;
}
