#include <filesystem>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

#ifdef USE_IRL
// Disable warnings for IRL
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <irl/geometry/general/pt.h>
#include <irl/interface_reconstruction_methods/elvira_neighborhood.h>
#include <irl/interface_reconstruction_methods/reconstruction_interface.h>
#pragma clang diagnostic pop
#endif  // USE_IRL

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

  auto dvofdt = make_centered();

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

  for (size_t i = 0; i < fs.vof.extent(0); ++i) {
    for (size_t j = 0; j < fs.vof.extent(1); ++j) {
      constexpr size_t nsample = 4;
      auto is_in               = [](Float y) -> Float { return y <= 0.5; };
      fs.vof[i, j]             = 0.0;
      for (size_t jj = 1; jj <= nsample; ++jj) {
        fs.vof[i, j] +=
            is_in(fs.y[j] + static_cast<Float>(jj) / static_cast<Float>(nsample + 1) * fs.dy[j]);
      }
      fs.vof[i, j] /= nsample;
    }
  }

  apply_velocity_bconds(fs);
  apply_vof_bconds(fs);

  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);
  calc_divergence(fs, div);
  if (!save_state(fs.x, fs.y, Ui, Vi, fs.p, div, fs.vof, t)) { return 1; }
  // = Initialize flow field =======================================================================

#ifdef USE_IRL
  // = Reconstruct the interface ===================================================================
  {
    constexpr size_t NUM_NEIGHBORS = 9;
    IRL::ELVIRANeighborhood neighborhood{};
    neighborhood.resize(NUM_NEIGHBORS);
    std::array<IRL::RectangularCuboid, NUM_NEIGHBORS> cells{};
    std::array<Float, NUM_NEIGHBORS> cells_vof{};

    size_t counter = 0;
    for (size_t i = 1; i < 4; ++i) {
      for (size_t j = 1; j < 4; ++j) {
        cells[counter] = IRL::RectangularCuboid::fromBoundingPts(
            IRL::Pt{fs.x[i], fs.y[j], -std::max(fs.dx[i], fs.dy[j]) / 2},
            IRL::Pt{fs.x[i + 1], fs.y[j + 1], std::max(fs.dx[i], fs.dy[j]) / 2});
        cells_vof[counter] = fs.vof[i, j];
        neighborhood.setMember(
            &cells[counter], &cells_vof[counter], static_cast<int>(i) - 2, static_cast<int>(j) - 2);
        counter += 1;
      }
    }
    const auto planar_separator = IRL::reconstructionWithELVIRA2D(neighborhood);
    IGOR_DEBUG_PRINT(planar_separator.getNumberOfPlanes());
    for (const auto& plane : planar_separator) {
      const auto norm = std::sqrt(Igor::sqr(plane.normal()[0]) + Igor::sqr(plane.normal()[1]) +
                                  Igor::sqr(plane.normal()[2]));
      IGOR_DEBUG_PRINT(norm);
      IGOR_DEBUG_PRINT(plane.normal());
      IGOR_DEBUG_PRINT(plane.distance());
    }
    for (size_t i = 0; i < fs.vof.extent(0); ++i) {
      for (size_t j = 0; j < fs.vof.extent(1); ++j) {
        std::cout << fs.vof[i, j] << '\t';
      }
      std::cout << '\n';
    }
  }
  Igor::Todo("Figure out IRL.");
  // = Reconstruct the interface ===================================================================
#endif  // USE_IRL

  Igor::ScopeTimer timer("Solver");
  bool failed = false;
  Igor::ProgressBar<Float> pbar(T_END, 67);
  while (t < T_END && !failed) {
    dt = adjust_dt(fs);
    dt = std::min(dt, T_END - t);

    // Save previous state
    std::copy_n(fs.U.get_data(), fs.U.size(), fs.U_old.get_data());
    std::copy_n(fs.V.get_data(), fs.V.size(), fs.V_old.get_data());
    std::copy_n(fs.vof.get_data(), fs.vof.size(), fs.vof_old.get_data());

    for (size_t sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);
      calc_mid_time(fs.vof, fs.vof_old);

      // = Update VOF field ========================================================================
      // calc_dvofdt(fs, dvofdt);
      // for (size_t i = 0; i < fs.vof.extent(0); ++i) {
      //   for (size_t j = 0; j < fs.vof.extent(1); ++j) {
      //     fs.vof[i, j] = fs.vof_old[i, j] + dt * dvofdt[i, j];
      //   }
      // }
      // apply_vof_bconds(fs);

      // = Update flow field =======================================================================
      // TODO: Handle density and interfaces
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
      apply_velocity_bconds(fs);

      calc_divergence(fs, div);
      // TODO: Add capillary forces here.
      if (!ps.solve(fs, div, dt, delta_p)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
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
      if (!save_state(fs.x, fs.y, Ui, Vi, fs.p, div, fs.vof, t)) { return 1; }
    }
    pbar.update(dt);
  }
  std::cout << '\n';

  if (failed) {
    Igor::Warn("Solver did not finish successfully.");
    return 1;
  } else {
    Igor::Info("Solver finish successfully.");
  }
}
