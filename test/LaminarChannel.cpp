#include <cstddef>
#include <filesystem>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"

#include "Common.hpp"

// = Config ========================================================================================
using Float = double;

constexpr Index NX = 500;
constexpr Index NY = 51;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 100.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr Float T_END    = 60.0;
constexpr Float DT_MAX   = 1e-1;
constexpr Float CFL_MAX  = 0.9;
constexpr Float DT_WRITE = 1.0;

constexpr Float U_IN   = 1.0;
constexpr Float U_INIT = 1.0;
constexpr Float VISC   = 1e-3;
constexpr Float RHO    = 0.5;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_IN, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "test/output/LaminarChannel/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  calc_rho_and_visc(Matrix<Float, NX, NY>{}, fs);

  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

  Float t  = 0.0;
  Float dt = DT_MAX;
  // = Allocate memory =============================================================================

  // = Initialize grid =============================================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * dx;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }
  init_mid_and_delta(fs);

  PS<Float, NX, NY> ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  std::fill_n(fs.p.get_data(), fs.p.size(), 0.0);

  for (Index i = 0; i < fs.U.extent(0); ++i) {
    for (Index j = 0; j < fs.U.extent(1); ++j) {
      fs.U[i, j] = U_INIT;
    }
  }
  for (Index i = 0; i < fs.V.extent(0); ++i) {
    for (Index j = 0; j < fs.V.extent(1); ++j) {
      fs.V[i, j] = 0.0;
    }
  }

  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);
  calc_divergence(fs, div);
  if (!save_state(OUTPUT_DIR, fs.x, fs.y, Ui, Vi, fs.p, div, /*fs.vof,*/ t)) { return 1; }
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("LaminarChannel");
  bool failed = false;
  while (t < T_END && !failed) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    std::copy_n(fs.U.get_data(), fs.U.size(), fs.U_old.get_data());
    std::copy_n(fs.V.get_data(), fs.V.size(), fs.V_old.get_data());

    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);

      // = Update flow field =======================================================================
      // TODO: Handle density and interfaces
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for (Index i = 0; i < fs.U.extent(0); ++i) {
        for (Index j = 0; j < fs.U.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.U[i, j] = fs.U_old[i, j] + dt * drhoUdt[i, j] / RHO;
        }
      }
      for (Index i = 0; i < fs.V.extent(0); ++i) {
        for (Index j = 0; j < fs.V.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.V[i, j] = fs.V_old[i, j] + dt * drhoVdt[i, j] / RHO;
        }
      }

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      calc_divergence(fs, div);
      // TODO: Add capillary forces here.
      if (!ps.solve(fs, div, dt, delta_p)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
      }

      shift_pressure_to_zero(fs, delta_p);
      for (Index i = 0; i < fs.p.extent(0); ++i) {
        for (Index j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      for (Index i = 1; i < fs.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.U.extent(1) - 1; ++j) {
          fs.U[i, j] -= (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / RHO;
        }
      }
      for (Index i = 1; i < fs.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.V.extent(1) - 1; ++j) {
          fs.V[i, j] -= (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / RHO;
        }
      }
    }

    t += dt;
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!save_state(OUTPUT_DIR, fs.x, fs.y, Ui, Vi, fs.p, div, /*fs.vof,*/ t)) { return 1; }
    }
  }

  if (failed) {
    Igor::Warn("LaminarChannel failed.");
    return 1;
  }

  // = Perform tests ===============================================================================
  bool any_test_failed = false;

  const auto i_above_60 = static_cast<Index>(std::find_if(fs.x.get_data(),
                                                          fs.x.get_data() + fs.x.size(),
                                                          [](Float xi) { return xi > 60.0; }) -
                                             fs.x.get_data());

  // Test pressure
  {
    constexpr Float TOL = 1e-4;
    for (Index i = i_above_60; i < fs.p.extent(0); ++i) {
      const auto ref_pressure = fs.p[i, 0];
      for (Index j = 0; j < fs.p.extent(1); ++j) {
        if (std::abs(fs.p[i, j] - ref_pressure) > TOL) {
          Igor::Warn("Non constant pressure along y-axis for x={}.", fs.xm[i]);
          any_test_failed = true;
        }
      }
    }

    const auto ref_dpdx =
        (fs.p[i_above_60 + 1, NY / 2] - fs.p[i_above_60, NY / 2]) / fs.dx[i_above_60 + 1];
    for (Index i = i_above_60 + 1; i < fs.p.extent(0); ++i) {
      const auto dpdx = (fs.p[i, NY / 2] - fs.p[i - 1, NY / 2]) / fs.dx[i];
      if (std::abs(ref_dpdx - dpdx) > TOL) {
        Igor::Warn("Non constant dpdx after x=60: Reference dpdx(x={})={:.6e}, dpdx(x={})={:.6e} "
                   "=> error = {:.6e}",
                   fs.x[i_above_60 + 1],
                   ref_dpdx,
                   fs.x[i],
                   dpdx,
                   std::abs(ref_dpdx - dpdx));
      }
    }
  }

  // Test U profile
  {
    constexpr Float TOL = 1e-2;
    auto u_analytical   = [](Float y, Float dpdx) -> Float {
      return dpdx / (2 * VISC) * (y * y - y);
    };
    Vector<Float, NY> diff{};

    constexpr size_t N_CHECKS                      = 3;
    constexpr std::array<size_t, N_CHECKS> i_check = {3 * NX / 4, 7 * NX / 8, NX - 1};
    std::array<Float, N_CHECKS> L1_errors{};

    size_t counter = 0;
    for (size_t i : i_check) {
      for (Index j = 0; j < fs.U.extent(1); ++j) {
        const auto dpdx = (fs.p[static_cast<Index>(i), j] - fs.p[static_cast<Index>(i - 1), j]) /
                          fs.dx[static_cast<Index>(i)];
        diff[j] = std::abs(fs.U[static_cast<Index>(i), j] - u_analytical(fs.ym[j], dpdx));
      }
      L1_errors[counter++] = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
    }

    counter = 0;
    for (size_t i : i_check) {
      const auto err = L1_errors[counter++];
      if (err > TOL) {
        Igor::Warn("U-velocity profile at x={} does not align with analytical solution: L1-error "
                   "is {:.6e}",
                   fs.x[static_cast<Index>(i)],
                   err);
        any_test_failed = true;
      }
    }
  }

  return any_test_failed ? 1 : 0;
}
