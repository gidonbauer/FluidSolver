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
#include <Igor/TypeName.hpp>

#include "Container.hpp"
#include "IO.hpp"

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 5;
constexpr Index NY    = 5;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

// Uniform velocity field
constexpr Float U  = 1.0;
constexpr Float V  = 0.5;
constexpr Float DT = 1e-2;

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
  IRL::PlanarSeparator separator{};
  const auto interface_filename = Igor::detail::format("{}/interface.txt", OUTPUT_DIR);
  std::ofstream interface_out(interface_filename);
  if (!interface_out) {
    Igor::Warn("Could not open file `{}`: {}", interface_filename, std::strerror(errno));
    return 1;
  }
  for (Index i = 1; i < vof.extent(0) - 1; ++i) {
    for (Index j = 1; j < vof.extent(1) - 1; ++j) {
      if (vof[i, j] < 1e-8 || vof[i, j] > (1 - 1e-8)) { continue; }

      IRL::ELVIRANeighborhood neighborhood{};
      neighborhood.resize(NEIGHBORHOOD_SIZE);
      std::array<IRL::RectangularCuboid, NEIGHBORHOOD_SIZE> cells{};
      std::array<Float, NEIGHBORHOOD_SIZE> cells_vof{};

      size_t counter = 0;
      for (Index di = -1; di <= 1; ++di) {
        for (Index dj = -1; dj <= 1; ++dj) {
          cells[counter] = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt{x[i + di], y[j + dj], -std::max(dx, dy) / 2},
              IRL::Pt{x[i + di + 1], y[j + dj + 1], std::max(dx, dy) / 2});
          cells_vof[counter] = vof[i + di, j + dj];
          neighborhood.setMember(&cells[counter], &cells_vof[counter], di, dj);
          counter += 1;
        }
      }
      const auto planar_separator = IRL::reconstructionWithELVIRA2D(neighborhood);
      if (i == 1 && j == 2) { separator = planar_separator; }
      IGOR_ASSERT(planar_separator.getNumberOfPlanes() == 1,
                  "({}, {}): Expected one planar but got {}",
                  i,
                  j,
                  planar_separator.getNumberOfPlanes());

      interface_out << i << ", " << j << ":\n";
      interface_out << "  normal: " << planar_separator[0].normal() << '\n';
      interface_out << "  distance: " << planar_separator[0].distance() << '\n';
    }
  }
  // = Reconstruct the interface ===================================================================

  // = Advect cell (0,2) ===========================================================================
  IRL::Dodecahedron advected_cell{};
  IRL::UnsignedIndex_t counter = 0;
  for (Index di = 0; di <= 1; ++di) {
    for (Index dj = 0; dj <= 1; ++dj) {
      for (Index dk = 0; dk <= 1; ++dk) {
        advected_cell[counter++] = IRL::Pt(x[0 + di] - DT * U - DT * V,
                                           y[2 + dj] - DT * U - DT * V,
                                           (2.0 * dk - 1.0) * std::min(dx, dy));
      }
    }
  }
  IGOR_DEBUG_PRINT(advected_cell.getNumberOfVertices());

  const auto vol = IRL::getVolumeMoments<IRL::Volume>(advected_cell, separator);
  IGOR_DEBUG_PRINT(Igor::type_name(vol));
  IGOR_DEBUG_PRINT(static_cast<double>(vol));
  IGOR_DEBUG_PRINT(Igor::type_name(advected_cell.calculateVolume()));
  IGOR_DEBUG_PRINT(static_cast<double>(advected_cell.calculateVolume()));
  // = Advect cell (0,2) ===========================================================================

  if (!to_npy(Igor::detail::format("{}/x.npy", OUTPUT_DIR), x)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/y.npy", OUTPUT_DIR), y)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/xm.npy", OUTPUT_DIR), xm)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/ym.npy", OUTPUT_DIR), ym)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/vof.npy", OUTPUT_DIR), vof)) { return 1; }
}
