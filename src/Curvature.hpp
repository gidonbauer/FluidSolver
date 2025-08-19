#ifndef FLUID_SOLVER_CURVATURE_HPP_
#define FLUID_SOLVER_CURVATURE_HPP_

#include <numbers>

#include <Igor/Math.hpp>
#include <Igor/StaticVector.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IR.hpp"
#include "Operators.hpp"
#include "Utility.hpp"
#include "VOF.hpp"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
namespace detail {

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void smooth_vf_field(const Vector<Float, NX, NGHOST>& xm,
                     const Vector<Float, NY, NGHOST>& ym,
                     const Matrix<Float, NX, NY, NGHOST>& vf,
                     Matrix<Float, NX, NY, NGHOST>& vf_smooth) noexcept {
  // They used 4 in the paper but 16 seems to work better for me
  constexpr Index NUM_SMOOTHING_CELLS = 4;  // 16;
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
      vf_smooth[i, j] = 0.0;
      for (Index di = -NUM_SMOOTHING_CELLS; di <= NUM_SMOOTHING_CELLS; ++di) {
        for (Index dj = -NUM_SMOOTHING_CELLS; dj <= NUM_SMOOTHING_CELLS; ++dj) {
          if (vf.is_valid_index(i + di, j + dj)) {
            vf_smooth[i, j] += vf[i + di, j + dj] *
                               smoothing_kernel(sqr_distance(xm[i], ym[j], xm[i + di], ym[j + dj]));
          }
        }
      }
    }
  }
}

constexpr size_t STATIC_STORAGE_CAPACITY = 100;

// -------------------------------------------------------------------------------------------------
enum : size_t { X, Y, NDIMS };
template <typename Float>
struct Interface {
  std::array<Float, NDIMS> begin{};
  std::array<Float, NDIMS> end{};
  std::array<Float, NDIMS> normal{};
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
auto extract_interface(Index i,
                       Index j,
                       const FS<Float, NX, NY, NGHOST>& fs,
                       const InterfaceReconstruction<NX, NY>& ir) -> Interface<Float> {
  IGOR_ASSERT((ir.interface[i, j].getNumberOfPlanes() == 1),
              "Expected exactly one plane but got {}",
              (ir.interface[i, j].getNumberOfPlanes()));
  const auto& plane    = ir.interface[i, j][0];
  const auto intersect = get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, plane);

  IGOR_ASSERT(std::abs(plane.normal()[NDIMS]) < 1e-12,
              "Expected z-component of normal to be zero but is {:.6e}",
              plane.normal()[NDIMS]);
  return {
      .begin  = {intersect[0][X], intersect[0][Y]},
      .end    = {intersect[1][X], intersect[1][Y]},
      .normal = {plane.normal()[X], plane.normal()[Y]},
  };
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t CAPACITY>
void rotate_interfaces(Igor::StaticVector<Interface<Float>, CAPACITY>& interfaces) {
  IGOR_ASSERT(
      interfaces.size() >= 1, "Expected at least one interface but got {}", interfaces.size());

  auto angle = std::acos(-interfaces[0].normal[Y]);
  if (interfaces[0].normal[X] > 0.0) { angle = 2.0 * std::numbers::pi - angle; }

  const std::array<Float, NDIMS> center_of_rotation{
      (interfaces[0].end[X] + interfaces[0].begin[X]) / 2.0,
      (interfaces[0].end[Y] + interfaces[0].begin[Y]) / 2.0,
  };

  auto rotate_vector = [angle](const std::array<Float, NDIMS>& vec) -> std::array<Float, NDIMS> {
    return {
        std::cos(angle) * vec[X] - std::sin(angle) * vec[Y],
        std::sin(angle) * vec[X] + std::cos(angle) * vec[Y],
    };
  };

  auto rotate_point = [angle, &center_of_rotation](
                          const std::array<Float, NDIMS>& vec) -> std::array<Float, NDIMS> {
    return {
        std::cos(angle) * (vec[X] - center_of_rotation[X]) -
            std::sin(angle) * (vec[Y] - center_of_rotation[Y]),
        std::sin(angle) * (vec[X] - center_of_rotation[X]) +
            std::cos(angle) * (vec[Y] - center_of_rotation[Y]),
    };
  };

  for (auto& i : interfaces) {
    i.begin  = rotate_point(i.begin);
    i.end    = rotate_point(i.end);
    i.normal = rotate_vector(i.normal);
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t CAPACITY>
void sort_begin_end(Igor::StaticVector<Interface<Float>, CAPACITY>& interfaces) {
  for (auto& i : interfaces) {
    if (i.begin[X] > i.end[X]) { std::swap(i.begin, i.end); }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float>
auto calc_cuvature_of_quad_poly(Float x, const Vector<Float, 3>& c) {
  const auto first_der  = c[1] + 2.0 * c[2] * x;
  const auto second_der = 2.0 * c[2];
  return second_der / std::pow(1.0 + Igor::sqr(first_der), 1.5);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t CAPACITY>
auto calc_curv_quad_volume_matching_impl(
    const Igor::StaticVector<Interface<Float>, CAPACITY>& interfaces) -> Float {
  Igor::StaticVector<std::array<Float, NDIMS>, STATIC_STORAGE_CAPACITY> plic_param{};
  for (const auto& i : interfaces) {
    const auto b1 = (i.end[Y] - i.begin[Y]) / (i.end[X] - i.begin[X]);
    const auto b0 = i.begin[Y] - b1 * i.begin[X];
    plic_param.push_back({b0, b1});
  }

  Igor::StaticVector<std::array<Float, 3UZ>, STATIC_STORAGE_CAPACITY> S{};
  for (const auto& i : interfaces) {
    S.push_back({
        i.end[X] - i.begin[X],
        1.0 / 2.0 * (std::pow(i.end[X], 2.0) - std::pow(i.begin[X], 2.0)),
        1.0 / 3.0 * (std::pow(i.end[X], 3.0) - std::pow(i.begin[X], 3.0)),
    });
  }

  Matrix<Float, 3, 3> A{};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      for (size_t r = 0; r < interfaces.size(); ++r) {
        A[i, j] += S[r][static_cast<size_t>(i)] * S[r][static_cast<size_t>(j)];
      }
    }
  }

  Vector<Float, 3> d{};
  for (Index i = 0; i < 3; ++i) {
    for (size_t r = 0; r < interfaces.size(); ++r) {
      d[i] +=
          S[r][static_cast<size_t>(i)] * (plic_param[r][0] * S[r][0] + plic_param[r][1] * S[r][1]);
    }
  }

  Vector<Float, 3> c{};
  solve_linear_system(A, d, c);
  IGOR_ASSERT(
      std::abs(interfaces[0].begin[Y] - interfaces[0].end[Y]) < 1e-8,
      "Expected target interface to be a horizontal line but begin[Y] = {:.6e} and end[Y] = {:.6e}",
      interfaces[0].begin[Y],
      interfaces[0].end[Y]);

  const auto eval_point = (interfaces[0].begin[X] + interfaces[0].end[X]) / 2.0;
  return calc_cuvature_of_quad_poly(eval_point, c);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t CAPACITY>
void solve_linear_least_squares(
    const Igor::StaticVector<std::array<Float, NDIMS>, CAPACITY>& points, Vector<Float, 3>& sol) {
  // c = (X^T * X)^(-1) * (X^T * y)

  Matrix<Float, 3, 3> A{};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      for (const auto [xi, _] : points) {
        A[i, j] += std::pow(xi, static_cast<Float>(i)) * std::pow(xi, static_cast<Float>(j));
      }
    }
  }

  Vector<Float, 3> b{};
  for (Index i = 0; i < 3; ++i) {
    for (const auto [xi, yi] : points) {
      b[i] += std::pow(xi, static_cast<Float>(i)) * yi;
    }
  }

  solve_linear_system(A, b, sol);
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
// Reference: A Paraboloid Fitting Technique for  Calculating Curvature from Piecewise-Linear
//            Interface Reconstructions on 3D  Unstructured Meshes by Z. Jibben, N. N. Carlson, and
//            M. M. Francois
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_curvature_quad_volume_matching(const FS<Float, NX, NY, NGHOST>& fs,
                                         VOF<Float, NX, NY, NGHOST>& vof) {
  using detail::Interface;
  using detail::STATIC_STORAGE_CAPACITY;
  using detail::X, detail::Y;

  for_each_i<Exec::Parallel>(vof.curv, [&](Index i, Index j) {
    if (has_interface(vof.vf_old, i, j)) {
      Igor::StaticVector<Interface<Float>, STATIC_STORAGE_CAPACITY> interfaces{};
      // Target interface is the first one
      interfaces.push_back(detail::extract_interface(i, j, fs, vof.ir));
      for (Index di = -1; di <= 1; ++di) {
        for (Index dj = -1; dj <= 1; ++dj) {
          const bool check_here = (di != 0 || dj != 0);
          if (check_here && vof.ir.interface.is_valid_index(i + di, j + dj) &&
              has_interface(vof.vf_old, i + di, j + dj)) {
            interfaces.push_back(detail::extract_interface(i + di, j + dj, fs, vof.ir));
          }
        }
      }

      rotate_interfaces(interfaces);
      // IGOR_ASSERT(std::abs(interfaces[0].normal[X]) < 1e-10 &&
      //                 std::abs(interfaces[0].normal[Y] + 1.0) < 1e-10,
      //             "Expected normal of target interface to point in direction (0, -1) but is "
      //             "({:.6e}, {:.6e})",
      //             interfaces[0].normal[X],
      //             interfaces[0].normal[Y]);

      sort_begin_end(interfaces);
      vof.curv[i, j] = detail::calc_curv_quad_volume_matching_impl(interfaces);
    } else {
      vof.curv[i, j] = 0.0;
    }
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_curvature_quad_regression(const FS<Float, NX, NY, NGHOST>& fs,
                                    VOF<Float, NX, NY, NGHOST>& vof) {
  using detail::Interface;
  using detail::STATIC_STORAGE_CAPACITY;
  using detail::X, detail::Y, detail::NDIMS;

  for_each_i<Exec::Parallel>(vof.curv, [&](Index i, Index j) {
    if (has_interface(vof.vf_old, i, j)) {
      Igor::StaticVector<Interface<Float>, STATIC_STORAGE_CAPACITY> interfaces{};
      // Target interface is the first one
      interfaces.push_back(detail::extract_interface(i, j, fs, vof.ir));
      for (Index di = -1; di <= 1; ++di) {
        for (Index dj = -1; dj <= 1; ++dj) {
          const bool check_here = (di != 0 || dj != 0);
          if (check_here && vof.ir.interface.is_valid_index(i + di, j + dj) &&
              has_interface(vof.vf_old, i + di, j + dj)) {
            interfaces.push_back(detail::extract_interface(i + di, j + dj, fs, vof.ir));
          }
        }
      }

      detail::rotate_interfaces(interfaces);
      IGOR_ASSERT(std::abs(interfaces[0].normal[X]) < 1e-10 &&
                      std::abs(interfaces[0].normal[Y] + 1.0) < 1e-10,
                  "Expected normal of target interface to point in direction (0, -1) but is "
                  "({:.6e}, {:.6e})",
                  interfaces[0].normal[X],
                  interfaces[0].normal[Y]);

      Igor::StaticVector<std::array<Float, NDIMS>, 3UZ * STATIC_STORAGE_CAPACITY> points{};
      for (const auto& [begin, end, _] : interfaces) {
        points.push_back({
            (begin[X] + end[X]) / 2.0,
            (begin[Y] + end[Y]) / 2.0,
        });
      }
      Vector<Float, 3> c{};
      detail::solve_linear_least_squares(points, c);
      vof.curv[i, j] = detail::calc_cuvature_of_quad_poly(points[0][X], c);
    } else {
      vof.curv[i, j] = 0.0;
    }
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_curvature_convolved_vf(const FS<Float, NX, NY, NGHOST>& fs,
                                 VOF<Float, NX, NY, NGHOST>& vof) noexcept {
  // Reference: Cummins, S. J., Francois, M. M., and Kothe, D. B. “Estimating curvature from volume
  // fractions”. Computers & Structures. Frontier of Multi-Phase Flow Analysis and
  // Fluid-StructureFrontier of MultiPhase Flow Analysis and Fluid-Structure 83.6 (2005), pp.
  // 425–434.

  static Matrix<Float, NX, NY, NGHOST> dvfdx{};
  static Matrix<Float, NX, NY, NGHOST> dvfdy{};
  static Matrix<Float, NX, NY, NGHOST> dvfdxx{};
  static Matrix<Float, NX, NY, NGHOST> dvfdyy{};
  static Matrix<Float, NX, NY, NGHOST> dvfdxy{};
  static Matrix<Float, NX, NY, NGHOST> vf_smooth{};
  static Matrix<Float, NX, NY, NGHOST> curv_centered{};

  detail::smooth_vf_field(fs.xm, fs.ym, vof.vf_old, vf_smooth);

  calc_grad_of_centered_points(vf_smooth, fs.dx, fs.dy, dvfdx, dvfdy);
  calc_grad_of_centered_points(dvfdx, fs.dx, fs.dy, dvfdxx, dvfdxy);
  calc_grad_of_centered_points(dvfdy, fs.dx, fs.dy, dvfdxy, dvfdyy);

  for_each_i<Exec::Parallel>(curv_centered, [&](Index i, Index j) {
    const auto numer =
        (dvfdxx[i, j] * Igor::sqr(dvfdy[i, j]) + dvfdyy[i, j] * Igor::sqr(dvfdx[i, j]) -
         2.0 * dvfdx[i, j] * dvfdy[i, j] * dvfdxy[i, j]);
    const auto denom    = std::pow(Igor::sqr(dvfdx[i, j]) + Igor::sqr(dvfdy[i, j]), 1.5);
    curv_centered[i, j] = std::abs(denom) > 1e-8 ? -numer / denom : 0.0;
  });

#ifdef FS_CURV_NO_INTERPOLATION
  for_each_i<Exec::Parallel>(vof.curv, [&](Index i, Index j) {
    if (has_interface(vof.vf_old, i, j)) {
      vof.curv[i, j] = curv_centered[i, j];
    } else {
      vof.curv[i, j] = 0.0;
    }
  });
#else
  for_each_i<Exec::Parallel>(vof.curv, [&](Index i, Index j) {
    if (has_interface(vof.vf_old, i, j)) {
      const auto intersect =
          get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, vof.ir.interface[i, j][0]);
      const auto center = (intersect[0] + intersect[1]) / 2.0;
      vof.curv[i, j]    = bilinear_interpolate(fs.xm, fs.ym, curv_centered, center[0], center[1]);
    } else {
      vof.curv[i, j] = 0.0;
    }
  });
#endif
}

// -------------------------------------------------------------------------------------------------
// template <typename Float, Index NX, Index NY, Index NGHOST>
// void calc_surface_length(const FS<Float, NX, NY>& fs,
//                          const InterfaceReconstruction<NX, NY>& ir,
//                          Matrix<Float, NX, NY>& surface_length) noexcept {
//   for (Index i = 0; i < NX; ++i) {
//     for (Index j = 0; j < NY; ++j) {
//       const auto& interface = ir.interface[i, j];
//       if (interface.getNumberOfPlanes() > 0) {
//         IGOR_ASSERT(interface.getNumberOfPlanes() == 1,
//                     "Expected exactly one plane but got {}",
//                     interface.getNumberOfPlanes());
//         surface_length[i, j] = get_interface_length<Float, NX, NY>(i, j, fs.x, fs.y,
//         interface[0]);
//       } else {
//         surface_length[i, j] = 0.0;
//       }
//     }
//   }
// }

#endif  // FLUID_SOLVER_CURVATURE_HPP_
