// Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
// fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
// Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp. 425–434.

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

// #define FS_CURV_NO_INTERPOLATION
#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                        = double;
constexpr Index NX                 = 16;
constexpr Index NY                 = 16;
constexpr Float X_MIN              = 0.0;
constexpr Float X_MAX              = 1.0;
constexpr Float Y_MIN              = 0.0;
constexpr Float Y_MAX              = 1.0;
[[maybe_unused]] constexpr auto DX = (X_MAX - X_MIN) / static_cast<Float>(NX);
[[maybe_unused]] constexpr auto DY = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float R                  = 0.25;
constexpr Float CX                 = 0.65;
constexpr Float CY                 = 0.35;
constexpr auto vof0(Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R));
  // return static_cast<Float>(Igor::sqr(x - 0.5) + Igor::sqr(2 * (y - 0.5)) <= Igor::sqr(R));
}

constexpr auto OUTPUT_DIR = "output/Curvature";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
#if 0
void curvature_from_smoothed_vof(const FS<Float, NX, NY>& fs,
                                 const InterfaceReconstruction<NX, NY>& ir,
                                 const Matrix<Float, NX, NY>& vof,
                                 Matrix<Float, NX, NY>& vof_smooth,
                                 Matrix<Float, NX, NY>& curv,
                                 Matrix<Float, NX, NY>& curv_interpolated,
                                 Matrix<Float, NX, NY>& dvofdx,
                                 Matrix<Float, NX, NY>& dvofdy,
                                 Matrix<Float, NX, NY>& dvofdxx,
                                 Matrix<Float, NX, NY>& dvofdyy,
                                 Matrix<Float, NX, NY>& dvofdxy) {
  // = Calculate curvature =========================================================================
  IGOR_TIME_SCOPE("Smoothing") { smooth_vof_field(fs.xm, fs.ym, vof, vof_smooth); }
  IGOR_TIME_SCOPE("Calculating the gradient") {
    calc_grad_of_centered_points(vof_smooth, DX, DY, dvofdx, dvofdy);
    calc_grad_of_centered_points(dvofdx, DX, DY, dvofdxx, dvofdxy);
    calc_grad_of_centered_points(dvofdy, DX, DY, dvofdxy, dvofdyy);
  }

  Float min_curv  = std::numeric_limits<Float>::max();
  Float max_curv  = -std::numeric_limits<Float>::max();
  Float mean_curv = 0.0;
  Float mse_curv  = 0.0;
  Index count     = 0;
  IGOR_TIME_SCOPE("Calculating the curvature") {
    std::fill_n(curv.get_data(), curv.size(), std::numeric_limits<Float>::quiet_NaN());

    // TODO: Find center of interface an interpolate curvture at that point
    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        const auto numer =
            (dvofdxx[i, j] * Igor::sqr(dvofdy[i, j]) + dvofdyy[i, j] * Igor::sqr(dvofdx[i, j]) -
             2.0 * dvofdx[i, j] * dvofdy[i, j] * dvofdxy[i, j]);
        const auto denom = std::pow(Igor::sqr(dvofdx[i, j]) + Igor::sqr(dvofdy[i, j]), 1.5);
        if (std::abs(denom) > 1e-8) {
          curv[i, j] = -numer / denom;
        } else {
          curv[i, j] = std::numeric_limits<Float>::quiet_NaN();
        }

        if (has_interface(vof, i, j)) {
          min_curv   = std::min(curv[i, j], min_curv);
          max_curv   = std::max(curv[i, j], max_curv);
          mean_curv += curv[i, j];
          mse_curv  += Igor::sqr(curv[i, j] - 1.0 / R);
          count     += 1;
        }
      }
    }

    mean_curv /= static_cast<Float>(count);
    mse_curv  /= static_cast<Float>(count);
  }
  Igor::Info("NX={}, NY={}", NX, NY);
  Igor::Info("Mean curvature     = {:.6e}", mean_curv);
  Igor::Info("Min. curvature     = {:.6e}", min_curv);
  Igor::Info("Max. curvature     = {:.6e}", max_curv);
  Igor::Info("Expected curvature = {:.6e}", 1.0 / R);
  Igor::Info("MSE curvature      = {:.6e}", mse_curv);
  // = Calculate curvature =========================================================================

  // = Interpolate curvature to interface center ===================================================
  std::fill_n(curv_interpolated.get_data(),
              curv_interpolated.size(),
              std::numeric_limits<Float>::quiet_NaN());

  min_curv  = std::numeric_limits<Float>::max();
  max_curv  = -std::numeric_limits<Float>::max();
  mean_curv = 0.0;
  mse_curv  = 0.0;
  count     = 0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        const auto intersect =
            get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, ir.interface[i, j][0]);
        const auto center        = (intersect[0] + intersect[1]) / 2.0;
        curv_interpolated[i, j]  = bilinear_interpolate(fs.xm, fs.ym, curv, center[0], center[1]);

        min_curv                 = std::min(curv[i, j], min_curv);
        max_curv                 = std::max(curv[i, j], max_curv);
        mean_curv               += curv_interpolated[i, j];
        mse_curv                += Igor::sqr(curv_interpolated[i, j] - 1.0 / R);
        count                   += 1;
      }
    }
  }
  mean_curv /= static_cast<Float>(count);
  mse_curv  /= static_cast<Float>(count);
  Igor::Info("Mean curvature (interpolated)     = {:.6e}", mean_curv);
  Igor::Info("Min. curvature (interpolated)     = {:.6e}", min_curv);
  Igor::Info("Max. curvature (interpolated)     = {:.6e}", max_curv);
  Igor::Info("Expected curvature (interpolated) = {:.6e}", 1.0 / R);
  Igor::Info("MSE curvature (interpolated)      = {:.6e}", mse_curv);
  // = Interpolate curvature to interface center ===================================================
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (!has_interface(vof, i, j)) { curv[i, j] = std::numeric_limits<Float>::quiet_NaN(); }
    }
  }
}
#endif

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
auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  InterfaceReconstruction<NX, NY> ir{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> smooth_curv{};

  Matrix<Float, NX, NY> poly_curv{};

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_scalar("smooth_curv", &smooth_curv);
  vtk_writer.add_scalar("poly_curv", &poly_curv);
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, ir);
  // = Setup grid and cell localizers ==============================================================

  // = Initialize VOF field ========================================================================
  IGOR_TIME_SCOPE("Initializing VOF") {
    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        vof[i, j] = quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx * fs.dy);
      }
    }
  }
  reconstruct_interface(fs.x, fs.y, vof, ir);
  // = Initialize VOF field ========================================================================

  IGOR_TIME_SCOPE("Smoothed curvature calculation") { calc_curvature(fs, ir, vof, smooth_curv); }

  constexpr Float expected_curv = 1.0 / R;
  Float min_curv                = std::numeric_limits<Float>::max();
  Float max_curv                = -std::numeric_limits<Float>::max();
  Float mean_curv               = 0.0;
  Float mse_curv                = 0.0;
  Index count                   = 0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        min_curv   = std::min(smooth_curv[i, j], min_curv);
        max_curv   = std::max(smooth_curv[i, j], max_curv);
        mean_curv += smooth_curv[i, j];
        mse_curv  += Igor::sqr(smooth_curv[i, j] - expected_curv);
        count     += 1;
      }
    }
  }
  mean_curv /= static_cast<Float>(count);
  mse_curv  /= static_cast<Float>(count);
  Igor::Info("Mean curvature     = {:.6e}", mean_curv);
  Igor::Info("Min. curvature     = {:.6e}", min_curv);
  Igor::Info("Max. curvature     = {:.6e}", max_curv);
  Igor::Info("Expected curvature = {:.6e}", expected_curv);
  Igor::Info("MSE curvature      = {:.6e}", mse_curv);

  if (!vtk_writer.write()) { return 1; }
  if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, 0),
                      fs.x,
                      fs.y,
                      ir.interface)) {
    return 1;
  }
  monitor.write();

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
