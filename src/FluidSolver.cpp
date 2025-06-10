#include <filesystem>
#include <mdspan>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/MdArray.hpp>
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
  FS fs = {
      .x  = make_nx_p1(),
      .xm = make_nx(),
      .dx = make_nx(),

      .y  = make_ny_p1(),
      .ym = make_ny(),
      .dy = make_ny(),

      .U     = make_u_staggered(),
      .U_old = make_u_staggered(),

      .V     = make_v_staggered(),
      .V_old = make_v_staggered(),

      .p = make_centered(),
  };

  auto Ui  = make_centered();
  auto Vi  = make_centered();
  auto div = make_centered();

  auto resU    = make_u_staggered();
  auto resV    = make_v_staggered();
  auto delta_p = make_centered();

  Float t  = 0.0;
  Float dt = DT_MAX;
  // = Allocate memory =============================================================================

  // = Initialize HYPRE for pressure correction ====================================================
  PS ps{};
  // = Initialize HYPRE for pressure correction ====================================================

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
  bool quit = false;
  while (t < T_END && !quit) {
    dt = adjust_dt(fs, dt);
    dt = std::min(dt, T_END - t);

    // Save previous state
    fs.U_old = fs.U;
    fs.V_old = fs.V;

    for (size_t sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);

      calc_dmomdt(fs, resU, resV);
      for (size_t i = 0; i < fs.U.extent(0); ++i) {
        for (size_t j = 0; j < fs.U.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.U[i, j] = fs.U_old[i, j] + dt * resU[i, j] / RHO;
        }
      }
      for (size_t i = 0; i < fs.V.extent(0); ++i) {
        for (size_t j = 0; j < fs.V.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.V[i, j] = fs.V_old[i, j] + dt * resV[i, j] / RHO;
        }
      }

      // Boundary conditions
      apply_bconds(fs);

      calc_divergence(fs, div);
      if (!ps.solve(fs, div, dt, delta_p)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        quit = true;
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
    ts.push_back(t);
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    if (!save_state(fs.x, fs.y, Ui, Vi, fs.p, div, t)) { return 1; }
  }

  if (!Igor::mdspan_to_npy(std::mdspan(ts.data(), ts.size()),
                           Igor::detail::format("{}/t.npy", OUTPUT_DIR))) {
    return 1;
  }
}
