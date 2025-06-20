// Disable warnings for IRL
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include <irl/geometry/general/pt.h>
#include <irl/interface_reconstruction_methods/elvira_neighborhood.h>
#include <irl/interface_reconstruction_methods/reconstruction_interface.h>
#pragma clang diagnostic pop

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Container.hpp"
#include "IO.hpp"

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 10;
constexpr Index NY    = 10;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr size_t NEIGHBORHOOD_SIZE = 9;

constexpr auto OUTPUT_DIR = "output/VOF";
// = Config ========================================================================================

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
  Matrix<Float, NX, NY> vof{};
  Vector<Float, NX + 1> x{};
  Vector<Float, NY + 1> y{};
  Vector<Float, NX> xm{};
  Vector<Float, NY> ym{};
  // = Allocate memory =============================================================================

  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);
  for (Index i = 0; i < x.extent(0); ++i) {
    x[i] = X_MIN + static_cast<Float>(i) * dx;
  }
  for (Index i = 0; i < xm.extent(0); ++i) {
    xm[i] = (x[i] + x[i + 1]) / 2.0;
  }
  for (Index j = 0; j < y.extent(0); ++j) {
    y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }
  for (Index j = 0; j < ym.extent(0); ++j) {
    ym[j] = (y[j] + y[j + 1]) / 2.0;
  }

  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      constexpr Index nsample = 4;
      auto is_in              = [](Float x, Float y) -> Float {
        return Igor::sqr(x - 0.5) + Igor::sqr(y - 0.5) <= Igor::sqr(0.25);
      };

      vof[i, j] = 0.0;
      for (Index ii = 1; ii <= nsample; ++ii) {
        for (Index jj = 1; jj <= nsample; ++jj) {
          const auto xi = x[i] + static_cast<Float>(ii) / static_cast<Float>(nsample + 1) * dx;
          const auto yj = y[j] + static_cast<Float>(jj) / static_cast<Float>(nsample + 1) * dy;
          vof[i, j] += is_in(xi, yj);
        }
      }
      vof[i, j] /= Igor::sqr(nsample);
    }
  }

  // = Reconstruct the interface ===================================================================
  for (Index i = 1; i < vof.extent(0) - 1; ++i) {
    for (Index j = 1; j < vof.extent(1) - 1; ++j) {
      if (vof[i, j] < 1e-8 || vof[i, j] > (1 - 1e-8)) { continue; }

      IRL::ELVIRANeighborhood neighborhood{};
      neighborhood.resize(NEIGHBORHOOD_SIZE);
      std::array<IRL::RectangularCuboid, NEIGHBORHOOD_SIZE> cells{};
      std::array<Float, NEIGHBORHOOD_SIZE> cells_vof{};

      size_t counter = 0;
      for (int ii = -1; ii <= 1; ++ii) {
        for (int jj = -1; jj <= 1; ++jj) {
          cells[counter] = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt{x[i + ii], y[j + jj], -std::max(dx, dy) / 2},
              IRL::Pt{x[i + ii + 1], y[j + jj + 1], std::max(dx, dy) / 2});
          cells_vof[counter] = vof[i, j];
          neighborhood.setMember(&cells[counter], &cells_vof[counter], ii, jj);
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
    }
  }

  // = Reconstruct the interface ===================================================================

  if (!to_npy(Igor::detail::format("{}/xm.npy", OUTPUT_DIR), xm)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/ym.npy", OUTPUT_DIR), ym)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/vof.npy", OUTPUT_DIR), vof)) { return 1; }
}
