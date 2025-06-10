#include <filesystem>
#include <mdspan>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/MdArray.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "Config.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"

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
  FS fs{};
  PS ps{};

  auto Ui  = make_centered();
  auto Vi  = make_centered();
  auto div = make_centered();

  auto drhoUdt = make_u_staggered();
  auto drhoVdt = make_v_staggered();
  auto delta_p = make_centered();

  Float t  = 0.0;
  Float dt = DT_MAX;
  // = Allocate memory =============================================================================

  // = Initialize grid =============================================================================
  for (size_t i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * (X_MAX - X_MIN) / static_cast<Float>(NX);
  }
  for (size_t i = 0; i < fs.xm.extent(0); ++i) {
    fs.xm[i] = (fs.x[i] + fs.x[i + 1]) / 2;
    fs.dx[i] = fs.x[i + 1] - fs.x[i];
  }
  for (size_t j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * (Y_MAX - Y_MIN) / static_cast<Float>(NY);
  }
  for (size_t j = 0; j < fs.ym.extent(0); ++j) {
    fs.ym[j] = (fs.y[j] + fs.y[j + 1]) / 2;
    fs.dy[j] = fs.y[j + 1] - fs.y[j];
  }
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  std::fill_n(fs.p.get_data(), fs.p.size(), 0.0);

  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    for (size_t j = 0; j < fs.U.extent(1); ++j) {
      fs.U[i, j] = U_IN;
    }
  }
  for (size_t i = 0; i < fs.V.extent(0); ++i) {
    for (size_t j = 0; j < fs.V.extent(1); ++j) {
      fs.V[i, j] = 0.0;
    }
  }
  apply_bconds(fs);

  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);
  calc_divergence(fs, div);
  if (!save_state(fs.x, fs.y, Ui, Vi, fs.p, div, t)) { return 1; }
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  std::vector<Float> ts;
  ts.push_back(t);
  bool failed = false;
  Igor::ProgressBar<Float> pbar(T_END, 60);
  while (t < T_END && !failed) {
    dt = adjust_dt(fs);
    dt = std::min(dt, T_END - t);

    // Save previous state
    fs.U_old = fs.U;
    fs.V_old = fs.V;

    for (size_t sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);

      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for (size_t i = 0; i < fs.U.extent(0); ++i) {
        for (size_t j = 0; j < fs.U.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.U[i, j] = fs.U_old[i, j] + dt * drhoUdt[i, j] / RHO;
        }
      }
      for (size_t i = 0; i < fs.V.extent(0); ++i) {
        for (size_t j = 0; j < fs.V.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.V[i, j] = fs.V_old[i, j] + dt * drhoVdt[i, j] / RHO;
        }
      }

      // Boundary conditions
      apply_bconds(fs);

      calc_divergence(fs, div);
      if (!ps.solve(fs, div, dt, delta_p)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        // failed = true;
      }

      shift_pressure_to_zero(fs, delta_p);
      for (size_t i = 0; i < fs.p.extent(0); ++i) {
        for (size_t j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      for (size_t i = 1; i < fs.U.extent(0) - 1; ++i) {
        for (size_t j = 1; j < fs.U.extent(1) - 1; ++j) {
          fs.U[i, j] -= (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / RHO;
        }
      }
      for (size_t i = 1; i < fs.V.extent(0) - 1; ++i) {
        for (size_t j = 1; j < fs.V.extent(1) - 1; ++j) {
          fs.V[i, j] -= (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / RHO;
        }
      }
    }

    t += dt;
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    if (should_save(t, dt)) {
      if (!save_state(fs.x, fs.y, Ui, Vi, fs.p, div, t)) { return 1; }
      ts.push_back(t);
    }
    pbar.update(dt);
  }
  std::cout << '\n';

  if (!Igor::mdspan_to_npy(std::mdspan(ts.data(), ts.size()),
                           Igor::detail::format("{}/t.npy", OUTPUT_DIR))) {
    return 1;
  }

  if (failed) {
    Igor::Warn("Solver did not finish successfully.");
    return 1;
  } else {
    Igor::Info("Solver finish successfully.");
  }
}
