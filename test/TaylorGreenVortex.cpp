#include <cstddef>
#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 128;
constexpr Index NY              = 128;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 2.0 * std::numbers::pi_v<Float>;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 2.0 * std::numbers::pi_v<Float>;

constexpr Float T_END           = 5.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-2;

constexpr Float VISC            = 1e-1;
constexpr Float RHO             = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

constexpr auto OUTPUT_DIR       = "test/output/TaylorGreenVortex/";

constexpr FlowBConds<Float> bconds{
    //        LEFT             RIGHT            BOTTOM           TOP
    .types = {BCond::PERIODIC, BCond::PERIODIC, BCond::PERIODIC, BCond::PERIODIC},
    .U     = {},
    .V     = {},
};
// = Config ========================================================================================

auto F(Float t) -> Float { return std::exp(-2.0 * VISC / RHO * t); }
auto u_analytical(Float x, Float y, Float t) -> Float { return std::sin(x) * std::cos(y) * F(t); }
auto v_analytical(Float x, Float y, Float t) -> Float { return -std::cos(x) * std::sin(y) * F(t); }

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

  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float U_max   = 0.0;
  Float V_max   = 0.0;
  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0.0;
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

  VTKWriter<Float, NX, NY, NGHOST> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::NONE);

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(
      fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = u_analytical(fs.x(i), fs.ym(j), 0.0); });
  for_each_i<Exec::Parallel>(
      fs.curr.U, [&](Index i, Index j) { fs.curr.V(i, j) = v_analytical(fs.xm(i), fs.y(j), 0.0); });
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

  Igor::ScopeTimer timer("TaylorGreenVortex");
  bool failed = false;
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
        failed = true;
      }
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for_each_a(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

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
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  if (failed) {
    Igor::Warn("TaylorGreenVortex failed.");
    return 1;
  }

  // = Perform tests ===============================================================================
  bool any_test_failed = false;
  const Float TOL      = 3.0 * Igor::sqr(std::max(fs.dx, fs.dy));

  const auto vol       = fs.dx * fs.dy;

  // Test U
  if (std::any_of(fs.curr.U.get_data(), fs.curr.U.get_data() + fs.curr.U.size(), [](Float value) {
        return std::isnan(value);
      })) {
    Igor::Warn("NaN value in U.");
    any_test_failed = true;
  }

  Float L1_error_U = 0.0;
  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      L1_error_U += std::abs(fs.curr.U(i, j) - u_analytical(fs.x(i), fs.ym(j), T_END)) * vol;
    }
  }
  if (L1_error_U > TOL) {
    Igor::Warn("U profile is incorrect: L1 error = {:.6e}", L1_error_U);
    any_test_failed = true;
  }

  // Test V
  if (std::any_of(fs.curr.V.get_data(), fs.curr.V.get_data() + fs.curr.V.size(), [](Float value) {
        return std::isnan(value);
      })) {
    Igor::Warn("NaN value in V.");
    any_test_failed = true;
  }

  Float L1_error_V = 0.0;
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      L1_error_V += std::abs(fs.curr.V(i, j) - v_analytical(fs.xm(i), fs.y(j), T_END)) * vol;
    }
  }
  if (L1_error_V > TOL) {
    Igor::Warn("V profile is incorrect: L1 error = {:.6e}", L1_error_V);
    any_test_failed = true;
  }

  return any_test_failed ? 1 : 0;
}
