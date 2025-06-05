#include <cmath>
#include <filesystem>
#include <mdspan>

#include <omp.h>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/MdArray.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
// #define FS_USE_PFMG_PRECOND
#define FS_WALL_FULL_LENGTH
#define DEBUG_SAVE

#include "Config.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  omp_set_num_threads(1);

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

      .rho     = make_centered(),
      .rho_old = make_centered(),

      .U     = make_u_staggered(),
      .U_old = make_u_staggered(),

      .V     = make_v_staggered(),
      .V_old = make_v_staggered(),

      .p    = make_centered(),
      .visc = make_centered(),
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

  // = Initialize HYPRE ============================================================================
  auto ps = init_pressure_correction();
  Igor::Defer fini_hypre([&ps] { finalize_pressure_correction(ps); });
  // = Initialize HYPRE ============================================================================

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

#ifdef DEBUG_SAVE
  if (!Igor::mdspan_to_npy(fs.x, Igor::detail::format("{}/x.npy", OUTPUT_DIR))) { return 1; }
  if (!Igor::mdspan_to_npy(fs.y, Igor::detail::format("{}/y.npy", OUTPUT_DIR))) { return 1; }
  if (!Igor::mdspan_to_npy(fs.xm, Igor::detail::format("{}/xm.npy", OUTPUT_DIR))) { return 1; }
  if (!Igor::mdspan_to_npy(fs.ym, Igor::detail::format("{}/ym.npy", OUTPUT_DIR))) { return 1; }
#endif
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  std::fill_n(fs.rho.get_data(), fs.rho.size(), RHO);
  std::fill_n(fs.p.get_data(), fs.p.size(), 0.0);
  std::fill_n(fs.visc.get_data(), fs.visc.size(), VISC);

  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    for (size_t j = 0; j < fs.U.extent(1); ++j) {
      // fs.U[i, j] = 0.0;
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
  if (!save_state(fs.x, fs.y, fs.rho, Ui, Vi, fs.p, div, t)) { return 1; }
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  std::vector<Float> ts;
  ts.push_back(t);
  while (t < T_END) {
    std::cout << "t = " << t << '\r' << std::flush;
    dt = adjust_dt(fs, dt);
    dt = std::min(dt, T_END - t);

    // Save previous state
    fs.rho_old = fs.rho;
    fs.U_old   = fs.U;
    fs.V_old   = fs.V;

    for (size_t sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);

      calc_dmomdt(fs, resU, resV);
#ifdef DEBUG_SAVE
      if (!Igor::mdspan_to_npy(
              resU, Igor::detail::format("{}/drhoUdt_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
      if (!Igor::mdspan_to_npy(
              resV, Igor::detail::format("{}/drhoVdt_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
#endif  // DEBUG_SAVE
      {
        if (std::any_of(resU.get_data(), resU.get_data() + resU.size(), [](auto value) {
              return std::isnan(value);
            })) {
          Igor::Warn("NaN value in resU.");
        }
        if (std::any_of(resV.get_data(), resV.get_data() + resV.size(), [](auto value) {
              return std::isnan(value);
            })) {
          Igor::Warn("NaN value in resV.");
        }
      }

      Float max_U_change = 0.0;
      Float max_V_change = 0.0;
      for (size_t i = 0; i < fs.U.extent(0); ++i) {
        for (size_t j = 0; j < fs.U.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.U[i, j]   = fs.U_old[i, j] + dt * resU[i, j] / fs.rho[i, j];
          fs.V[i, j]   = fs.V_old[i, j] + dt * resV[i, j] / fs.rho[i, j];
          max_U_change = std::max(max_U_change, std::abs(dt * resU[i, j] / fs.rho[i, j]));
          max_V_change = std::max(max_V_change, std::abs(dt * resV[i, j] / fs.rho[i, j]));
        }
      }
      Igor::Debug("Max. U change dmomdt = {}", max_U_change);
      Igor::Debug("Max. V change dmomdt = {}", max_V_change);
#ifdef DEBUG_SAVE
      if (!Igor::mdspan_to_npy(
              fs.U,
              Igor::detail::format("{}/U_pre_bcond_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
      if (!Igor::mdspan_to_npy(
              fs.V,
              Igor::detail::format("{}/V_pre_bcond_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
#endif  // DEBUG_SAVE

      // Boundary conditions
      apply_bconds(fs);
#ifdef DEBUG_SAVE
      if (!Igor::mdspan_to_npy(
              fs.U, Igor::detail::format("{}/U_pre_corr_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
      if (!Igor::mdspan_to_npy(
              fs.V, Igor::detail::format("{}/V_pre_corr_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
#endif  // DEBUG_SAVE

      calc_divergence(fs, div);
      if (!calc_pressure_correction(fs, div, ps, dt, delta_p)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        return 1;
      }

      shift_pressure_to_zero(fs, delta_p);
      for (size_t i = 0; i < fs.p.extent(0); ++i) {
        for (size_t j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      max_U_change = 0.0;
      max_V_change = 0.0;
      for (size_t i = 1; i < fs.U.extent(0) - 1; ++i) {
        for (size_t j = 1; j < fs.U.extent(1) - 1; ++j) {
          fs.U[i, j] += (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / fs.rho[i, j];
          fs.V[i, j] += (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / fs.rho[i, j];
          max_U_change = std::max(
              max_U_change,
              std::abs((delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / fs.rho[i, j]));
          max_V_change = std::max(
              max_V_change,
              std::abs((delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / fs.rho[i, j]));
        }
      }
      Igor::Debug("Max. U change pressure = {}", max_U_change);
      Igor::Debug("Max. V change pressure = {}", max_V_change);
#ifdef DEBUG_SAVE
      if (!Igor::mdspan_to_npy(
              fs.U,
              Igor::detail::format("{}/U_post_corr_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
      if (!Igor::mdspan_to_npy(
              fs.V,
              Igor::detail::format("{}/V_post_corr_{:.6f}_{}.npy", OUTPUT_DIR, t, sub_iter))) {
        return 1;
      }
#endif  // DEBUG_SAVE
    }

    t += dt;
    ts.push_back(t);
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    if (!save_state(fs.x, fs.y, fs.rho, Ui, Vi, fs.p, div, t)) { return 1; }
    // break;
  }

  if (!Igor::mdspan_to_npy(std::mdspan(ts.data(), ts.size()),
                           Igor::detail::format("{}/t.npy", OUTPUT_DIR))) {
    return 1;
  }
}
