#ifndef FLUID_SOLVER_CURVATURE_HPP_
#define FLUID_SOLVER_CURVATURE_HPP_

// Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
// fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
// Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp. 425–434.

#include <Igor/Math.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IR.hpp"
#include "Operators.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void smooth_vof_field(const Vector<Float, NX>& xm,
                      const Vector<Float, NY>& ym,
                      const Matrix<Float, NX, NY>& vof,
                      Matrix<Float, NX, NY>& vof_smooth) noexcept {
  // They used 4 in the paper but 16 seems to work better for me
  constexpr Index NUM_SMOOTHING_CELLS = 16;
  const Float SMOOTHING_LENGTH = NUM_SMOOTHING_CELLS * std::max(xm[1] - xm[0], ym[1] - ym[0]);
  const auto smoothing_kernel  = [SMOOTHING_LENGTH](Float sqr_distance) {
    IGOR_ASSERT(sqr_distance >= 0.0, "Squared-distance must be positive but is {}", sqr_distance);
    sqr_distance /= Igor::sqr(SMOOTHING_LENGTH);
    if (sqr_distance >= 1.0) { return 0.0; }
    return std::pow(1.0 - sqr_distance, 4.0);
  };

  constexpr auto sqr_distance = [](Float x1, Float y1, Float x2, Float y2) {
    return Igor::sqr(x2 - x1) + Igor::sqr(y2 - y1);
  };

#pragma omp parallel for collapse(2)
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      vof_smooth[i, j] = 0.0;
      for (Index di = -NUM_SMOOTHING_CELLS; di <= NUM_SMOOTHING_CELLS; ++di) {
        for (Index dj = -NUM_SMOOTHING_CELLS; dj <= NUM_SMOOTHING_CELLS; ++dj) {
          if (vof.is_valid_index(i + di, j + dj)) {
            vof_smooth[i, j] +=
                vof[i + di, j + dj] *
                smoothing_kernel(sqr_distance(xm[i], ym[j], xm[i + di], ym[j + dj]));
          }
        }
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_curvature(const FS<Float, NX, NY>& fs,
                    const InterfaceReconstruction<NX, NY>& ir,
                    const Matrix<Float, NX, NY>& vof,
                    Matrix<Float, NX, NY>& curv) noexcept {
  static Matrix<Float, NX, NY> dvofdx{};
  static Matrix<Float, NX, NY> dvofdy{};
  static Matrix<Float, NX, NY> dvofdxx{};
  static Matrix<Float, NX, NY> dvofdyy{};
  static Matrix<Float, NX, NY> dvofdxy{};
  static Matrix<Float, NX, NY> vof_smooth{};
  static Matrix<Float, NX, NY> curv_centered{};

  smooth_vof_field(fs.xm, fs.ym, vof, vof_smooth);

  calc_grad_of_centered_points(vof_smooth, fs.dx, fs.dy, dvofdx, dvofdy);
  calc_grad_of_centered_points(dvofdx, fs.dx, fs.dy, dvofdxx, dvofdxy);
  calc_grad_of_centered_points(dvofdy, fs.dx, fs.dy, dvofdxy, dvofdyy);

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      const auto numer =
          (dvofdxx[i, j] * Igor::sqr(dvofdy[i, j]) + dvofdyy[i, j] * Igor::sqr(dvofdx[i, j]) -
           2.0 * dvofdx[i, j] * dvofdy[i, j] * dvofdxy[i, j]);
      const auto denom    = std::pow(Igor::sqr(dvofdx[i, j]) + Igor::sqr(dvofdy[i, j]), 1.5);
      curv_centered[i, j] = std::abs(denom) > 1e-8 ? -numer / denom : 0.0;
    }
  }

#ifdef FS_CURV_NO_INTERPOLATION
  (void)ir;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        curv[i, j] = curv_centered[i, j];
      } else {
        curv[i, j] = 0.0;
      }
    }
  }
#else
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        const auto intersect =
            get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, ir.interface[i, j][0]);
        const auto center = (intersect[0] + intersect[1]) / 2.0;
        curv[i, j]        = bilinear_interpolate(fs.xm, fs.ym, curv_centered, center[0], center[1]);
      } else {
        curv[i, j] = 0.0;
      }
    }
  }
#endif
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_surface_length(const FS<Float, NX, NY>& fs,
                         const InterfaceReconstruction<NX, NY>& ir,
                         Matrix<Float, NX, NY>& surface_length) noexcept {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      const auto& interface = ir.interface[i, j];
      if (interface.getNumberOfPlanes() > 0) {
        IGOR_ASSERT(interface.getNumberOfPlanes() == 1,
                    "Expected exactly one plane but got {}",
                    interface.getNumberOfPlanes());
        surface_length[i, j] = get_interface_length<Float, NX, NY>(i, j, fs.x, fs.y, interface[0]);
      } else {
        surface_length[i, j] = 0.0;
      }
    }
  }
}
#endif  // FLUID_SOLVER_CURVATURE_HPP_
