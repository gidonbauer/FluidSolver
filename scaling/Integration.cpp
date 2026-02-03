#include <iomanip>
#include <iostream>
#include <numbers>

#include "Quadrature.hpp"

// =================================================================================================
#if 1
auto f(double x) -> double { return x * x * std::sin(x) * std::sin(x) * std::sin(x); }
constexpr double X_MIN             = 0.0;
constexpr double X_MAX             = 2.0 * std::numbers::pi;
constexpr double INTEGRAL_EXPECTED = -8.0 * Igor::sqr(std::numbers::pi) / 3.0;
#else
auto f(double x) -> double { return std::exp(-x * x); }
constexpr double X_MIN         = -2.0;
constexpr double X_MAX         = 2.0;
const double INTEGRAL_EXPECTED = Igor::sqrt(std::numbers::pi) * std::erf(2.0);
#endif

// =================================================================================================
template <size_t N>
void check_quadrature() {
  const double integral  = quadrature<N>(f, X_MIN, X_MAX);
  const double abs_error = std::abs(integral - INTEGRAL_EXPECTED);
  const double rel_error = abs_error / std::abs(INTEGRAL_EXPECTED);

  std::cout << std::scientific << std::setprecision(6) << "Quadrature" << ',' << N << ','
            << integral << ',' << INTEGRAL_EXPECTED << ',' << abs_error << ',' << rel_error << '\n';

  if constexpr (N + 1 <= detail::MAX_QUAD_N) { check_quadrature<N + 1>(); }
}

// =================================================================================================
void check_simpson_rule(size_t N) {
  std::vector<double> fs(N);
  for (size_t i = 0; i < N; ++i) {
    const double x = X_MIN + static_cast<double>(i) * (X_MAX - X_MIN) / static_cast<double>(N - 1);
    fs[i]          = f(x);
  }
  const auto integral    = simpsons_rule(std::span(fs.data(), fs.size()), X_MIN, X_MAX);
  const double abs_error = std::abs(integral - INTEGRAL_EXPECTED);
  const double rel_error = abs_error / std::abs(INTEGRAL_EXPECTED);

  std::cout << std::scientific << std::setprecision(6) << "SimpsonsRule" << ',' << N << ','
            << integral << ',' << INTEGRAL_EXPECTED << ',' << abs_error << ',' << rel_error << '\n';
}

// =================================================================================================
void check_midpoint_rule(size_t N) {
  const double dx = (X_MAX - X_MIN) / static_cast<double>(N);
  std::vector<double> fs(N);
  for (size_t i = 0; i < N; ++i) {
    const double xm = X_MIN + (static_cast<double>(i) + 0.5) * dx;
    fs[i]           = f(xm);
  }
  const auto integral    = midpoint_rule(std::span(fs.data(), fs.size()), dx);
  const double abs_error = std::abs(integral - INTEGRAL_EXPECTED);
  const double rel_error = abs_error / std::abs(INTEGRAL_EXPECTED);

  std::cout << std::scientific << std::setprecision(6) << "MidpointRule" << ',' << N << ','
            << integral << ',' << INTEGRAL_EXPECTED << ',' << abs_error << ',' << rel_error << '\n';
}

// =================================================================================================
void check_trapezoidal_rule(size_t N) {
  std::vector<double> xs(N);
  std::vector<double> fs(N);
  const double dx = (X_MAX - X_MIN) / static_cast<double>(N - 1);
  for (size_t i = 0; i < N; ++i) {
    xs[i] = X_MIN + static_cast<double>(i) * dx;
    fs[i] = f(xs[i]);
  }
  const auto integral =
      trapezoidal_rule(std::span(fs.data(), fs.size()), std::span(xs.data(), xs.size()));
  const double abs_error = std::abs(integral - INTEGRAL_EXPECTED);
  const double rel_error = abs_error / std::abs(INTEGRAL_EXPECTED);

  std::cout << std::scientific << std::setprecision(6) << "TrapezoidalRule" << ',' << N << ','
            << integral << ',' << INTEGRAL_EXPECTED << ',' << abs_error << ',' << rel_error << '\n';
}

// =================================================================================================
auto main() -> int {
  std::cout << "Method,N,Result,Expected,AbsError,RelError\n";

  check_quadrature<1>();

  for (size_t N = 3; N < 11; N += 2) {
    check_simpson_rule(N);
    check_trapezoidal_rule(N);
    check_midpoint_rule(N);
  }
  for (size_t N = 11; N < 502; N += 10) {
    check_simpson_rule(N);
    check_trapezoidal_rule(N);
    check_midpoint_rule(N);
  }
}
