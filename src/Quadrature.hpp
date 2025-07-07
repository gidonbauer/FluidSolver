#ifndef FLUID_SOLVER_QUADRATURE_
#define FLUID_SOLVER_QUADRATURE_

#include <cmath>
#include <numeric>

#include <Igor/Math.hpp>

#include "QuadratureTables.hpp"

namespace detail {

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto quadrature(FUNC f,
                                        const std::array<std::array<Float, 2>, 4>& domain) noexcept
    -> Float {
  enum : size_t { X, Y };
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::constexpr_abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                                    static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  // Jacobian matrix of coordinate transform
  // J11
  auto dxdxi = [&domain](Float eta) -> Float {
    return 1.0 / 4.0 *
           ((domain[1][X] - domain[0][X]) * (1 - eta) + (domain[2][X] - domain[3][X]) * (1 + eta));
  };
  // J12
  auto dydxi = [&domain](Float eta) -> Float {
    return 1.0 / 4.0 *
           ((domain[1][Y] - domain[0][Y]) * (1 - eta) + (domain[2][Y] - domain[3][Y]) * (1 + eta));
  };
  // J21
  auto dxdeta = [&domain](Float xi) -> Float {
    return 1.0 / 4.0 *
           ((domain[3][X] - domain[0][X]) * (1 - xi) + (domain[2][X] - domain[1][X]) * (1 + xi));
  };
  // J22
  auto dydeta = [&domain](Float xi) -> Float {
    return 1.0 / 4.0 *
           ((domain[3][Y] - domain[0][Y]) * (1 - xi) + (domain[2][Y] - domain[1][Y]) * (1 + xi));
  };

  auto abs_det_J = [&](Float xi, Float eta) -> Float {
    return std::abs(dxdxi(eta) * dydeta(xi) - dydxi(eta) * dxdeta(xi));
  };

  // Shape functions
  constexpr std::array<Float (*)(Float, Float), 4UZ> psi = {
      [](Float xi, Float eta) -> Float { return (1.0 / 4.0) * (1.0 - xi) * (1.0 - eta); },
      [](Float xi, Float eta) -> Float { return (1.0 / 4.0) * (1.0 + xi) * (1.0 - eta); },
      [](Float xi, Float eta) -> Float { return (1.0 / 4.0) * (1.0 + xi) * (1.0 + eta); },
      [](Float xi, Float eta) -> Float { return (1.0 / 4.0) * (1.0 - xi) * (1.0 + eta); },
  };

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    for (size_t yidx = 0; yidx < gauss_points.size(); ++yidx) {
      const auto xi  = gauss_points[xidx];
      const auto wx  = gauss_weights[xidx];
      const auto eta = gauss_points[yidx];
      const auto wy  = gauss_weights[yidx];

      auto x = static_cast<Float>(0);
      auto y = static_cast<Float>(0);
      for (size_t i = 0; i < psi.size(); ++i) {
        x += psi[i](xi, eta) * domain[i][X];
        y += psi[i](xi, eta) * domain[i][Y];
      }

      integral += wx * wy * f(x, y) * abs_det_J(xi, eta);
    }
  }
  return integral;
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto quadrature(FUNC f, Float x_min, Float x_max) noexcept -> Float {
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::constexpr_abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                                    static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    const auto xi = gauss_points[xidx];
    const auto w  = gauss_weights[xidx];

    const auto x = (x_max - x_min) / 2 * xi + (x_max + x_min) / 2;
    integral += w * f(x);
  }
  return (x_max - x_min) / 2 * integral;
}

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto
quadrature(FUNC f, Float x_min, Float x_max, Float y_min, Float y_max) noexcept -> Float {
  return detail::quadrature<N>(f,
                               std::array{
                                   std::array{x_min, y_min},
                                   std::array{x_max, y_min},
                                   std::array{x_max, y_max},
                                   std::array{x_min, y_max},
                               });
}

#endif  // FLUID_SOLVER_QUADRATURE_
