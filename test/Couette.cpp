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

constexpr Index NX = 110;
constexpr Index NY = 11;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 10.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr Float T_END    = 5.0;
constexpr Float DT_MAX   = 1e-1;
constexpr Float CFL_MAX  = 0.9;
constexpr Float DT_WRITE = 0.1;

constexpr Float U_TOP  = 1.0;
constexpr Float U_INIT = 0.0;
constexpr Float VISC   = 1e-1;
constexpr Float RHO    = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER = 2;

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
auto main() -> int {
  // = Create output directory =====================================================================
  {
    std::error_code ec;

    std::filesystem::remove_all(OUTPUT_DIR, ec);
    if (ec) {
      Igor::Warn("Could remove directory `{}`: {}", OUTPUT_DIR, ec.message());
      return 1;
    }

    std::filesystem::create_directories(OUTPUT_DIR, ec);
    if (ec) {
      Igor::Warn("Could not create directory `{}`: {}", OUTPUT_DIR, ec.message());
      return 1;
    }
  }
  // = Create output directory =====================================================================

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc = VISC, .rho = RHO};
  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);
  PS<Float, NX, NY> ps(dx, dy, PRESSURE_TOL, PRESSURE_MAX_ITER);

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
  for (Index i = 0; i < fs.xm.extent(0); ++i) {
    fs.xm[i] = (fs.x[i] + fs.x[i + 1]) / 2;
    fs.dx[i] = fs.x[i + 1] - fs.x[i];
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }
  for (Index j = 0; j < fs.ym.extent(0); ++j) {
    fs.ym[j] = (fs.y[j] + fs.y[j + 1]) / 2;
    fs.dy[j] = fs.y[j + 1] - fs.y[j];
  }
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

  Igor::ScopeTimer timer("Couette");
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
    Igor::Warn("Couette failed.");
    return 1;
  }

  // = Perform tests ===============================================================================
  bool any_test_failed = false;
  {
    auto u_analytical = [](Float y) -> Float { return U_TOP * y; };
    Vector<Float, NY> diff{};

    constexpr Index N_CHECKS                       = 3;
    constexpr std::array<size_t, N_CHECKS> i_check = {NX / 4, NX / 2, 3 * NX / 4};
    std::array<Float, N_CHECKS> L1_errors{};

    size_t counter = 0;
    for (size_t i : i_check) {
      for (Index j = 0; j < fs.U.extent(1); ++j) {
        diff[j] = std::abs(fs.U[static_cast<Index>(i), j] - u_analytical(fs.ym[j]));
      }
      L1_errors[counter++] = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
    }

    constexpr Float TOL = 1e-2;
    counter             = 0;
    for (size_t i : i_check) {
      const auto err = L1_errors[counter++];
      if (err > TOL) {
        Igor::Warn(
            "U-velocity profile at x={} does not align with analytical solution: L1-error is {}",
            fs.x[static_cast<Index>(i)],
            err);
        any_test_failed = true;
      }
    }
  }

  return any_test_failed ? 1 : 0;
}
