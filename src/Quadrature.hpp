#ifndef FLUID_SOLVER_QUADRATURE_
#define FLUID_SOLVER_QUADRATURE_

#include <numeric>

#include <Igor/Math.hpp>

#include "QuadratureTables.hpp"

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto quadrature(FUNC f, Float x_min, Float x_max) noexcept -> Float {
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                          static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    const auto xi  = gauss_points[xidx];
    const auto w   = gauss_weights[xidx];

    const auto x   = (x_max - x_min) / 2 * xi + (x_max + x_min) / 2;
    integral      += w * f(x);
  }
  return (x_max - x_min) / 2 * integral;
}

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto
quadrature(FUNC f, Float x_min, Float x_max, Float y_min, Float y_max) noexcept -> Float {
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                          static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    for (size_t yidx = 0; yidx < gauss_points.size(); ++yidx) {
      const auto wx    = gauss_weights[xidx];
      const auto wy    = gauss_weights[yidx];

      const auto xi_x  = gauss_points[xidx];
      const auto xi_y  = gauss_points[yidx];

      const auto x     = (x_max - x_min) / 2.0 * xi_x + (x_max + x_min) / 2.0;
      const auto y     = (y_max - y_min) / 2.0 * xi_y + (y_max + y_min) / 2.0;
      integral        += wx * wy * f(x, y);
    }
  }
  return (x_max - x_min) / 2.0 * (y_max - y_min) / 2.0 * integral;
}

#endif  // FLUID_SOLVER_QUADRATURE_
