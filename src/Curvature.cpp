#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 64;
constexpr Index NY    = 64;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;
constexpr auto DX     = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY     = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr auto vof0(Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - 0.5) + Igor::sqr(y - 0.5) <= Igor::sqr(0.25));
}

constexpr auto OUTPUT_DIR = "output/Curvature";
// = Config ========================================================================================

auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  InterfaceReconstruction<NX, NY> ir{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> curv{};

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_scalar("curv", &curv);
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * DX;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * DY;
  }
  init_mid_and_delta(fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, ir);
  // = Setup grid and cell localizers ==============================================================

  // = Initialize VOF field ========================================================================
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      vof[i, j] =
          quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx[i] * fs.dy[j]);
    }
  }
  reconstruct_interface(fs.x, fs.y, vof, ir);
  // = Initialize VOF field ========================================================================

  if (!vtk_writer.write()) { return 1; }
  if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, 0),
                      fs.x,
                      fs.y,
                      ir.interface)) {
    return 1;
  }
  monitor.write();
}
