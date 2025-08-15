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
using Float              = double;

constexpr Index NX       = 500;
constexpr Index NY       = 51;
constexpr Index NGHOST   = 1;

constexpr Float X_MIN    = 0.0;
constexpr Float X_MAX    = 100.0;
constexpr Float Y_MIN    = 0.0;
constexpr Float Y_MAX    = 1.0;

constexpr Float T_END    = 60.0;
constexpr Float DT_MAX   = 1e-1;
constexpr Float CFL_MAX  = 0.9;
constexpr Float DT_WRITE = 1.0;

constexpr Float U_IN     = 1.0;
#ifndef LC_U_INIT
constexpr Float U_INIT = 1.0;
#else
static_assert(
    std::is_convertible_v<std::remove_cvref_t<decltype(LC_U_INIT)>, Float>,
    "LC_U_INIT must have a value (the initital U-velocity) that must be convertible to Float.");
constexpr Float U_INIT = LC_U_INIT;
#endif
constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 0.5;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_IN, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

#ifndef LC_U_INIT
constexpr auto OUTPUT_DIR = "test/output/LaminarChannel/";
#else
constexpr auto OUTPUT_DIR = "test/output/LaminarChannel_" IGOR_STRINGIFY(LC_U_INIT) "/";
#endif
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0;
  outflow = 0;
  for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
    inflow  += fs.curr.rho_u_stag[-NGHOST, j] * fs.curr.U[-NGHOST, j];
    outflow += fs.curr.rho_u_stag[NX + NGHOST, j] * fs.curr.U[NX + NGHOST, j];
  }
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
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);
  // = Initialize pressure solver ==================================================================

  // = Initialize flow field =======================================================================
  for_each_i(fs.curr.U, [&](Index i, Index j) { fs.curr.U[i, j] = U_INIT; });
  for_each_i(fs.curr.V, [&](Index i, Index j) { fs.curr.V[i, j] = 0.0; });
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

#ifndef LC_U_INIT
  Igor::ScopeTimer timer("LaminarChannel");
#else
  Igor::ScopeTimer timer("LaminarChannel_" IGOR_STRINGIFY(LC_U_INIT));
#endif
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
      for_each_i(fs.curr.U, [&](Index i, Index j) {
        if (std::isnan(drhoUdt[i, j])) { Igor::Panic("NaN value in drhoUdt[{}, {}]", i, j); }
        fs.curr.U[i, j] = (fs.old.rho_u_stag[i, j] * fs.old.U[i, j] + dt * drhoUdt[i, j]) /
                          fs.curr.rho_u_stag[i, j];
      });
      for_each_i(fs.curr.V, [&](Index i, Index j) {
        if (std::isnan(drhoVdt[i, j])) { Igor::Panic("NaN value in drhoVdt[{}, {}]", i, j); }
        fs.curr.V[i, j] = (fs.old.rho_v_stag[i, j] * fs.old.V[i, j] + dt * drhoVdt[i, j]) /
                          fs.curr.rho_v_stag[i, j];
      });

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
        fs.curr.U[NX + NGHOST, j] -=
            mass_error / (fs.curr.rho_u_stag[NX + NGHOST, j] * static_cast<Float>(NY + 2 * NGHOST));
      }

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
      for_each_a(fs.p, [&](Index i, Index j) { fs.p[i, j] += delta_p[i, j]; });

      for_each_i(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U[i, j] -=
            (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx * dt / fs.curr.rho_u_stag[i, j];
      });
      for_each_i(fs.curr.V, [&](Index i, Index j) {
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
  const auto i_above_60 = static_cast<Index>(std::find_if(fs.x.get_data(),
                                                          fs.x.get_data() + fs.x.size(),
                                                          [](Float xi) { return xi > 60.0; }) -
                                             fs.x.get_data());

  // Test pressure
  {
    constexpr Float TOL = 1e-4;
    for (Index i = i_above_60; i < NX; ++i) {
      const auto ref_pressure = fs.p[i, 0];
      bool constant_pressure  = true;
      for (Index j = 0; j < fs.p.extent(1); ++j) {
        if (std::abs(fs.p[i, j] - ref_pressure) > TOL) { constant_pressure = false; }
      }
      if (!constant_pressure) {
        Igor::Warn("Non constant pressure along y-axis for x={}.", fs.xm[i]);
        any_test_failed = true;
      }
    }

    const auto ref_dpdx = (fs.p[i_above_60 + 1, NY / 2] - fs.p[i_above_60, NY / 2]) / fs.dx;
    for (Index i = i_above_60 + 1; i < fs.p.extent(0); ++i) {
      const auto dpdx = (fs.p[i, NY / 2] - fs.p[i - 1, NY / 2]) / fs.dx;
      if (std::abs(ref_dpdx - dpdx) > TOL) {
        Igor::Warn(
            "Non constant dpdx after x=60: Reference dpdx(x={:.6e})={:.6e}, dpdx(x={:.6e})={:.6e} "
            "=> error = {:.6e}",
            fs.x[i_above_60 + 1],
            ref_dpdx,
            fs.x[i],
            dpdx,
            std::abs(ref_dpdx - dpdx));
        any_test_failed = true;
      }
    }
  }

  // Test U profile
  {
    constexpr Float TOL = 2e-4;
    auto u_analytical   = [&](Float y, Float dpdx) -> Float {
      // NOTE: Adjustment due to the ghost cells, the dirichlet boundary condition is now enforced
      //       in the ghost cell
      return dpdx / (2 * VISC) * (y * y - y - (fs.dy / 2.0 + Igor::sqr(fs.dy / 2.0)));
      // return dpdx / (2 * VISC) * (y * y - y);
    };
    Vector<Float, NY + 2 * NGHOST, 0> diff{};

    static_assert(X_MIN == 0.0, "Expected X_MIN to be 0 to make things a bit easier.");
    constexpr Float TEST_X_BEGIN = 60.0;
    constexpr Float TEST_X_STEP  = 10.0;
    constexpr auto N_CHECKS      = static_cast<size_t>((X_MAX - TEST_X_BEGIN) / TEST_X_STEP);
    for (size_t i_check = 0; i_check < N_CHECKS; ++i_check) {
      const Float x_target = TEST_X_BEGIN + static_cast<Float>(i_check) * TEST_X_STEP;
      const auto i         = static_cast<Index>(x_target / X_MAX * static_cast<Float>(NX + 1));

      for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
        const auto dpdx = (fs.p[i, j] - fs.p[i - 1, j]) / fs.dx;
        diff[j + NGHOST] =
            std::abs(fs.curr.U[static_cast<Index>(i), j] - u_analytical(fs.ym[j], dpdx));
      }
      const auto L1_error = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
      if (L1_error > TOL) {
        Igor::Warn("U-velocity profile at x={} does not align with analytical solution: L1-error "
                   "is {:.6e}",
                   fs.x[i],
                   L1_error);
        any_test_failed = true;
      }
    }
  }

  return any_test_failed ? 1 : 0;
}
