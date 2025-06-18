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

// = Config ========================================================================================
using Float           = double;
constexpr size_t NX   = 3;
constexpr size_t NY   = 3;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

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
  // = Allocate memory =============================================================================

  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);
  for (size_t i = 0; i < x.extent(0); ++i) {
    x[i] = X_MIN + static_cast<Float>(i) * dx;
  }
  for (size_t j = 0; j < y.extent(0); ++j) {
    y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }

  for (size_t i = 0; i < vof.extent(0); ++i) {
    for (size_t j = 0; j < vof.extent(1); ++j) {
      constexpr size_t nsample = 4;
      auto is_in               = [](Float y) -> Float { return y <= 0.5; };
      vof[i, j]                = 0.0;
      for (size_t jj = 1; jj <= nsample; ++jj) {
        vof[i, j] += is_in(y[j] + static_cast<Float>(jj) / static_cast<Float>(nsample + 1) * dy);
      }
      vof[i, j] /= nsample;
    }
  }

  // = Reconstruct the interface ===================================================================
  constexpr size_t NUM_NEIGHBORS = 9;
  IRL::ELVIRANeighborhood neighborhood{};
  neighborhood.resize(NUM_NEIGHBORS);
  std::array<IRL::RectangularCuboid, NUM_NEIGHBORS> cells{};
  std::array<Float, NUM_NEIGHBORS> cells_vof{};

  size_t counter = 0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      cells[counter] = IRL::RectangularCuboid::fromBoundingPts(
          IRL::Pt{x[i], y[j], -std::max(dx, dy) / 2},
          IRL::Pt{x[i + 1], y[j + 1], std::max(dx, dy) / 2});
      cells_vof[counter] = vof[i, j];
      neighborhood.setMember(
          &cells[counter], &cells_vof[counter], static_cast<int>(i) - 1, static_cast<int>(j) - 1);
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
  for (size_t i = 0; i < vof.extent(0); ++i) {
    for (size_t j = 0; j < vof.extent(1); ++j) {
      std::cout << vof[i, j] << '\t';
    }
    std::cout << '\n';
  }
  // = Reconstruct the interface ===================================================================
}
