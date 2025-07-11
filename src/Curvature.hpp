#ifndef FLUID_SOLVER_CURVATURE_HPP_
#define FLUID_SOLVER_CURVATURE_HPP_

// Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
// fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
// Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp. 425–434.

#include <Igor/Math.hpp>

#include "Container.hpp"
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
  const auto smoothing_kernel  = [SMOOTHING_LENGTH](Float distance) {
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

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_curvature(Float dx,
                    Float dy,
                    const Matrix<Float, NX, NY>& vof,
                    const Matrix<Float, NX, NY>& vof_smooth,
                    Matrix<Float, NX, NY>& curv) noexcept {
  static Matrix<Float, NX, NY> dvofdx{};
  static Matrix<Float, NX, NY> dvofdy{};
  static Matrix<Float, NX, NY> dvofdxx{};
  static Matrix<Float, NX, NY> dvofdyy{};
  static Matrix<Float, NX, NY> dvofdxy{};

  calc_grad_of_centered_points(vof_smooth, dx, dy, dvofdx, dvofdy);
  calc_grad_of_centered_points(dvofdx, dx, dy, dvofdxx, dvofdxy);
  calc_grad_of_centered_points(dvofdy, dx, dy, dvofdxy, dvofdyy);

  std::fill_n(curv.get_data(), curv.size(), std::numeric_limits<Float>::quiet_NaN());

  // TODO: Maybe find center of interface an interpolate curvture at that point
  for (Index i = 1; i < NX - 1; ++i) {
    for (Index j = 1; j < NY - 1; ++j) {
      if (has_interface_in_neighborhood(vof, i, j, 2)) {
        curv[i, j] =
            (dvofdxx[i, j] * Igor::sqr(dvofdy[i, j]) + dvofdyy[i, j] * Igor::sqr(dvofdx[i, j]) -
             2.0 * dvofdx[i, j] * dvofdy[i, j] * dvofdxy[i, j]) /
            std::pow(Igor::sqr(dvofdx[i, j]) + Igor::sqr(dvofdy[i, j]), 1.5);
      }
    }
  }
}

#endif  // FLUID_SOLVER_CURVATURE_HPP_
