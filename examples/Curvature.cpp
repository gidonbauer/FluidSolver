// Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
// fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
// Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp. 425–434.

#include <random>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

// #define FS_CURV_NO_INTERPOLATION
#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                   = double;
constexpr Index NX            = 128;
constexpr Index NY            = 128;
constexpr Float X_MIN         = 0.0;
constexpr Float X_MAX         = 1.0;
constexpr Float Y_MIN         = 0.0;
constexpr Float Y_MAX         = 1.0;

constexpr Index NUM_TEST_ITER = 10000;

constexpr auto OUTPUT_DIR     = "output/Curvature";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void curvature_from_quad_reconstruction(const FS<Float, NX, NY>& fs,
                                        const InterfaceReconstruction<NX, NY>& ir,
                                        const Matrix<Float, NX, NY>& vof,
                                        Matrix<Float, NX, NY>& poly_curv) {
  (void)poly_curv;
  {
    const Index i = 3 * NX / 4;
    Index j       = 1;
    while (!has_interface(vof, i, j)) {
      j += 1;
    }

    Igor::Debug("{}, {}: normal = {}", i, j, ir.interface[i, j][0].normal());

    for (Index di = -1; di <= 1; ++di) {
      for (Index dj = -1; dj <= 1; ++dj) {
        if (has_interface(vof, i + di, j + dj)) {
          const auto intersects = get_intersections_with_cell<Float, NX, NY>(
              i + di, j + dj, fs.x, fs.y, ir.interface[i + di, j + dj][0]);
          const IRL::Pt center = 0.5 * (intersects[0] + intersects[1]);

          Igor::Debug(
              "{}, {}: center = ({}, {}, {})", i + di, j + dj, center[0], center[1], center[2]);
        }
      }
    }
  }

  std::cout << "all_p = np.array([";
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        const auto intersects =
            get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, ir.interface[i, j][0]);
        const IRL::Pt center = 0.5 * (intersects[0] + intersects[1]);
        std::cout << "                  [" << center[0] << ", " << center[1] << "],\n";
      }
    }
  }
  std::cout << "])\n";

  Igor::Todo("Curvature calculation using a quadratic reconstruction.");
}

// -------------------------------------------------------------------------------------------------
struct CurvatureMetrics {
  Float expected_curv{};
  Float min_curv{};
  Float max_curv{};
  Float mean_curv{};
  Float mse_curv{};
  Float mrse_curv{};
};

// -------------------------------------------------------------------------------------------------
void test_curvature(Float cx,
                    Float cy,
                    Float r,
                    const FS<Float, NX, NY>& fs,
                    InterfaceReconstruction<NX, NY>& ir,
                    Matrix<Float, NX, NY>& vof,
                    Matrix<Float, NX, NY>& curv,
                    CurvatureMetrics& metrics) {

  auto vof0 = [cx, cy, r](Float x, Float y) {
    return static_cast<Float>(Igor::sqr(x - cx) + Igor::sqr(y - cy) <= Igor::sqr(r));
  };
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      vof[i, j] = quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx * fs.dy);
    }
  }
  std::fill_n(ir.interface.get_data(), ir.interface.size(), IRL::PlanarSeparator{});
  reconstruct_interface(fs.x, fs.y, vof, ir);

  calc_curvature(fs, ir, vof, curv);

  metrics.expected_curv = 1.0 / r;
  metrics.min_curv      = std::numeric_limits<Float>::max();
  metrics.max_curv      = -std::numeric_limits<Float>::max();
  metrics.mean_curv     = 0.0;
  metrics.mse_curv      = 0.0;
  metrics.mrse_curv     = 0.0;
  Index count           = 0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        metrics.min_curv   = std::min(curv[i, j], metrics.min_curv);
        metrics.max_curv   = std::max(curv[i, j], metrics.max_curv);
        metrics.mean_curv += curv[i, j];
        metrics.mse_curv  += Igor::sqr(curv[i, j] - metrics.expected_curv);
        metrics.mrse_curv +=
            Igor::sqr(curv[i, j] - metrics.expected_curv) / Igor::sqr(metrics.expected_curv);
        count += 1;
      }
    }
  }
  metrics.mean_curv /= static_cast<Float>(count);
  metrics.mse_curv  /= static_cast<Float>(count);
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  InterfaceReconstruction<NX, NY> ir{};
  localize_cells(fs.x, fs.y, ir);

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> smooth_curv{};

  Matrix<Float, NX, NY> poly_curv{};

  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_scalar("smooth_curv", &smooth_curv);
  vtk_writer.add_scalar("poly_curv", &poly_curv);

  Index iter = 0;
  Float cx   = 0.0;
  Float cy   = 0.0;
  Float r    = 0.0;
  CurvatureMetrics smooth_metrics{};
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&iter, "iteration");
  monitor.add_variable(&cx, "center(x)");
  monitor.add_variable(&cy, "center(y)");
  monitor.add_variable(&r, "radius");
  monitor.add_variable(&smooth_metrics.expected_curv, "expect(curv)");
  monitor.add_variable(&smooth_metrics.min_curv, "min(curv)");
  monitor.add_variable(&smooth_metrics.max_curv, "max(curv)");
  monitor.add_variable(&smooth_metrics.mean_curv, "mean(curv)");
  monitor.add_variable(&smooth_metrics.mse_curv, "mse(curv)");
  monitor.add_variable(&smooth_metrics.mrse_curv, "mrse(curv)");
  // = Allocate memory =============================================================================

  static std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution c_dist(0.35, 0.65);
  std::uniform_real_distribution r_dist(0.01, 0.25);

  IGOR_TIME_SCOPE("Testing cuvature") {
    for (iter = 0; iter < NUM_TEST_ITER; ++iter) {
      cx = c_dist(generator);
      cy = c_dist(generator);
      r  = r_dist(generator);
      test_curvature(cx, cy, r, fs, ir, vof, smooth_curv, smooth_metrics);

      if (!vtk_writer.write()) { return 1; }
      if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                          fs.x,
                          fs.y,
                          ir.interface)) {
        return 1;
      }
      monitor.write();
    }
  }

  // if (!to_npy(Igor::detail::format("{}/xm.npy", OUTPUT_DIR), fs.xm)) { return 1; }
  // if (!to_npy(Igor::detail::format("{}/ym.npy", OUTPUT_DIR), fs.ym)) { return 1; }
  // if (!to_npy(Igor::detail::format("{}/VOF.npy", OUTPUT_DIR), vof)) { return 1; }

  // for (Index i = 0; i < ir.interface.extent(0); ++i) {
  //   for (Index j = 0; j < ir.interface.extent(1); ++j) {
  //     if (has_interface(vof, i, j)) {
  //       const auto [p0, p1] =
  //           get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, ir.interface[i, j][0]);
  //       std::cout << "p0 = " << p0 << '\n';
  //       std::cout << "p1 = " << p1 << '\n';
  //       std::cout << "n  = " << ir.interface[i, j][0].normal() << '\n';
  //       std::cout << '\n';
  //     }
  //   }
  // }

  // curvature_from_quad_reconstruction(fs, ir, vof, poly_curv);
}
