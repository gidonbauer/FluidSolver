// Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
// fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
// Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp. 425–434.

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 512;
constexpr Index NY    = 512;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;
constexpr auto DX     = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY     = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float R = 0.25;
constexpr auto vof0(Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - 0.5) + Igor::sqr(y - 0.5) <= Igor::sqr(R));
}

constexpr auto OUTPUT_DIR = "output/Curvature";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void smooth_vof_field(const Vector<Float, NX>& xm,
                      const Vector<Float, NY>& ym,
                      const Matrix<Float, NX, NY>& vof,
                      Matrix<Float, NX, NY>& vof_smooth) noexcept {
  // They used 4 in the paper but 16 seems to work better for me
  constexpr Index NUM_SMOOTHING_CELLS = 16;
  constexpr Float SMOOTHING_LENGTH    = NUM_SMOOTHING_CELLS * std::max(DX, DY);
  constexpr auto smoothing_kernel     = [](Float distance) {
    IGOR_ASSERT(distance >= 0.0, "Distance must be positive but is {}", distance);
    distance /= SMOOTHING_LENGTH;
    if (distance >= 1.0) { return 0.0; }
    return std::pow(1.0 - Igor::sqr(distance), 4.0);
  };

  constexpr auto distance = [](Float x1, Float y1, Float x2, Float y2) {
    return std::sqrt(Igor::sqr(x2 - x1) + Igor::sqr(y2 - y1));
  };

#pragma omp parallel for collapse(2)
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      vof_smooth[i, j] = 0.0;
      for (Index di = -NUM_SMOOTHING_CELLS; di <= NUM_SMOOTHING_CELLS; ++di) {
        for (Index dj = -NUM_SMOOTHING_CELLS; dj <= NUM_SMOOTHING_CELLS; ++dj) {
          if (vof.is_valid_index(i + di, j + dj)) {
            vof_smooth[i, j] += vof[i + di, j + dj] *
                                smoothing_kernel(distance(xm[i], ym[j], xm[i + di], ym[j + dj]));
          }
        }
      }
    }
  }
}

auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  InterfaceReconstruction<NX, NY> ir{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_smooth{};
  Matrix<Float, NX, NY> curv{};
  Matrix<Float, NX, NY> curv_interpolated{};

  Matrix<Float, NX, NY> dvofdx{};
  Matrix<Float, NX, NY> dvofdy{};
  Matrix<Float, NX, NY> dvofdxx{};
  Matrix<Float, NX, NY> dvofdyy{};
  Matrix<Float, NX, NY> dvofdxy{};

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_scalar("VOF_smooth", &vof_smooth);
  vtk_writer.add_scalar("curv", &curv);
  vtk_writer.add_scalar("curv_interpolated", &curv_interpolated);

  vtk_writer.add_scalar("dVOFdx", &dvofdx);
  vtk_writer.add_scalar("dVOFdy", &dvofdy);
  vtk_writer.add_scalar("dVOFdxx", &dvofdxx);
  vtk_writer.add_scalar("dVOFdyy", &dvofdyy);
  vtk_writer.add_scalar("dVOFdxy", &dvofdxy);
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
  IGOR_TIME_SCOPE("Initializing VOF") {
#pragma omp parallel for collapse(2)
    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        vof[i, j] =
            quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx[i] * fs.dy[j]);
      }
    }
  }
  reconstruct_interface(fs.x, fs.y, vof, ir);
  // = Initialize VOF field ========================================================================

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
    for (Index i = 1; i < NX - 1; ++i) {
      for (Index j = 1; j < NY - 1; ++j) {
        if (has_interface_in_neighborhood(vof, i, j, 2)) {
          curv[i, j] =
              (dvofdxx[i, j] * Igor::sqr(dvofdy[i, j]) + dvofdyy[i, j] * Igor::sqr(dvofdx[i, j]) -
               2.0 * dvofdx[i, j] * dvofdy[i, j] * dvofdxy[i, j]) /
              std::pow(Igor::sqr(dvofdx[i, j]) + Igor::sqr(dvofdy[i, j]), 1.5);

          if (has_interface(vof, i, j)) {
            min_curv = std::min(curv[i, j], min_curv);
            max_curv = std::max(curv[i, j], max_curv);
            mean_curv += curv[i, j];
            mse_curv += Igor::sqr(curv[i, j] + 1.0 / R);
            count += 1;
          }
        }
      }
    }

    mean_curv /= static_cast<Float>(count);
    mse_curv /= static_cast<Float>(count);
  }
  Igor::Info("NX={}, NY={}", NX, NY);
  Igor::Info("Mean curvature     = {:.6e}", mean_curv);
  Igor::Info("Min. curvature     = {:.6e}", min_curv);
  Igor::Info("Max. curvature     = {:.6e}", max_curv);
  Igor::Info("Expected curvature = {:.6e}", -1.0 / R);
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
        const auto center       = (intersect[0] + intersect[1]) / 2.0;
        curv_interpolated[i, j] = bilinear_interpolate(fs.xm, fs.ym, curv, center[0], center[1]);

        min_curv = std::min(curv[i, j], min_curv);
        max_curv = std::max(curv[i, j], max_curv);
        mean_curv += curv_interpolated[i, j];
        mse_curv += Igor::sqr(curv_interpolated[i, j] + 1.0 / R);
        count += 1;
      }
    }
  }
  mean_curv /= static_cast<Float>(count);
  mse_curv /= static_cast<Float>(count);
  Igor::Info("Mean curvature (interpolated)     = {:.6e}", mean_curv);
  Igor::Info("Min. curvature (interpolated)     = {:.6e}", min_curv);
  Igor::Info("Max. curvature (interpolated)     = {:.6e}", max_curv);
  Igor::Info("Expected curvature (interpolated) = {:.6e}", -1.0 / R);
  Igor::Info("MSE curvature (interpolated)      = {:.6e}", mse_curv);

  // = Interpolate curvature to interface center ===================================================

  IGOR_TIME_SCOPE("Writing") {
    if (!vtk_writer.write()) { return 1; }
    if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, 0),
                        fs.x,
                        fs.y,
                        ir.interface)) {
      return 1;
    }
    monitor.write();
  }
}
