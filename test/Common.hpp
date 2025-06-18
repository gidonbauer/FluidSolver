#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include <cstddef>

#include "Container.hpp"

// - Simpson's rule to integrate a function in 1D --------------------------------------------------
template <typename Float, size_t N>
[[nodiscard]] constexpr auto
simpsons_rule_1d(const Vector<Float, N>& f, Float x_min, Float x_max) noexcept -> Float {
  static_assert(N > 0 && N % 2 == 1, "n must be an odd number larger than zero");
  Float res = 0;
  for (size_t i = 1; i <= (N - 1) / 2; ++i) {
    res += f[2 * i - 2] + 4 * f[2 * i - 1] + f[2 * i];
  }
  const auto dx = (x_max - x_min) / static_cast<Float>(N);
  return res * dx / 3;
}

#endif  // TEST_COMMON_HPP_
