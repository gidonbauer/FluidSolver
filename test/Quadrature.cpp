#include <numbers>

#include "Quadrature.hpp"

// =================================================================================================
constexpr auto double_eq(double a,
                         double b,
                         double tol = Igor::sqrt(std::numeric_limits<double>::epsilon())) -> bool {
  return Igor::abs(a - b) <= tol;
}

// =================================================================================================
template <size_t N, size_t N_MAX>
constexpr auto check_all(auto f, double i_exp, double x_min, double x_max) {
  const auto i = quadrature<N>(f, x_min, x_max);
  if (!double_eq(i, i_exp)) {
    Igor::Error(
        "Incorrect result for quadrature (N={}): expected {:.6e} but got {:.6e}", N, i_exp, i);
    return false;
  }

  if constexpr (N + 1 < N_MAX) {
    return check_all<N + 1, N_MAX>(f, i_exp, x_min, x_max);
  } else {
    return true;
  }
}

// =================================================================================================
template <size_t N, size_t N_MAX>
constexpr auto
check_all(auto f, double i_exp, double x_min, double x_max, double y_min, double y_max) {
  const auto i = quadrature<N>(f, x_min, x_max, y_min, y_max);
  if (!double_eq(i, i_exp)) {
    Igor::Error(
        "Incorrect result for quadrature (N={}): expected {:.6e} but got {:.6e}", N, i_exp, i);
    return false;
  }

  if constexpr (N < N_MAX) {
    return check_all<N + 1, N_MAX>(f, i_exp, x_min, x_max, y_min, y_max);
  } else {
    return true;
  }
}

// =================================================================================================
auto main() -> int {
  // -----------------------------------------------------------------------------------------------
  {
    auto f           = [](double x) { return x * x; };
    const auto i_exp = 2.0 / 3.0;
    if (!check_all<2, detail::MAX_QUAD_N>(f, i_exp, -1.0, 1.0)) { return 1; }
  }

  // -----------------------------------------------------------------------------------------------
  {
    // See: https://www.wolframalpha.com/input?i=integrate+x%5E2+sin%5E3+x+dx+from+0+to+2pi
    auto f           = [](double x) { return x * x * std::sin(x) * std::sin(x) * std::sin(x); };
    const auto i_exp = -8.0 * Igor::sqr(std::numbers::pi) / 3.0;
    if (!check_all<14, detail::MAX_QUAD_N>(f, i_exp, 0.0, 2.0 * std::numbers::pi)) { return 1; }
  }

  // -----------------------------------------------------------------------------------------------
  {
    auto f = [](double x, double y) {
      return Igor::sqr(x) + std::sin(y) * std::sin(y) * std::sin(y);
    };
    const auto i_exp = 32.0 * std::numbers::pi / 3.0;
    if (!check_all<2, detail::MAX_QUAD_N>(f, i_exp, -2.0, 2.0, 0.0, 2.0 * std::numbers::pi)) {
      return 1;
    }
  }

  // - Simpsons rule -------------------------------------------------------------------------------
  {
    // See: https://www.wolframalpha.com/input?i=integrate+x%5E2+sin%5E3+x+dx+from+0+to+2pi
    auto f             = [](double x) { return x * x * std::sin(x) * std::sin(x) * std::sin(x); };
    const double x_min = 0.0;
    const double x_max = 2.0 * std::numbers::pi;
    const auto i_exp   = -8.0 * Igor::sqr(std::numbers::pi) / 3.0;

    for (size_t N = 11; N < 1002; N += 10) {
      std::vector<double> fs(N);
      for (size_t i = 0; i < fs.size(); ++i) {
        const double x =
            x_min + static_cast<double>(i) * (x_max - x_min) / static_cast<double>(N - 1);
        fs[i] = f(x);
      }
      const auto i = simpsons_rule(std::span(fs.data(), fs.size()), x_min, x_max);

      const double expected_error =
          (N > 100 ? 5000.0 : 10000.0) / static_cast<double>(Igor::sqr(Igor::sqr(N)));
      const double error = std::abs(i - i_exp);
      if (error > expected_error) {
        Igor::Error(
            "Incorrect result for Simpsons rule (N={}): expected error of <{:.6e} but got {:.6e}",
            N,
            expected_error,
            error);
        return 1;
      }
    }
  }

  // - Trapezoidal rule ----------------------------------------------------------------------------
  {
    // See: https://www.wolframalpha.com/input?i=integrate+x%5E2+sin%5E3+x+dx+from+0+to+2pi
    auto f             = [](double x) { return x * x * std::sin(x) * std::sin(x) * std::sin(x); };
    const double x_min = 0.0;
    const double x_max = 2.0 * std::numbers::pi;
    const auto i_exp   = -8.0 * Igor::sqr(std::numbers::pi) / 3.0;

    for (size_t N = 11; N < 1002; N += 10) {
      std::vector<double> xs(N);
      std::vector<double> fs(N);
      for (size_t i = 0; i < fs.size(); ++i) {
        xs[i] = x_min + static_cast<double>(i) * (x_max - x_min) / static_cast<double>(N - 1);
        fs[i] = f(xs[i]);
      }
      const auto i =
          trapezoidal_rule(std::span(fs.data(), fs.size()), std::span(xs.data(), xs.size()));

      const double expected_error = 5000.0 / static_cast<double>(Igor::sqr(Igor::sqr(N)));
      const double error          = std::abs(i - i_exp);
      if (error > expected_error) {
        Igor::Error("Incorrect result for Trapezoidal rule (N={}): expected error of <{:.6e} but "
                    "got {:.6e}",
                    N,
                    expected_error,
                    error);
        return 1;
      }
    }
  }

  // - Trapezoidal rule ----------------------------------------------------------------------------
  {
    // See: https://www.wolframalpha.com/input?i=integrate+x%5E2+sin%5E3+x+dx+from+0+to+2pi
    auto f             = [](double x) { return x * x * std::sin(x) * std::sin(x) * std::sin(x); };
    const double x_min = 0.0;
    const double x_max = 2.0 * std::numbers::pi;
    const auto i_exp   = -8.0 * Igor::sqr(std::numbers::pi) / 3.0;

    for (size_t N = 11; N < 1002; N += 10) {
      std::vector<double> fs(N);
      const double dx = (x_max - x_min) / static_cast<double>(N);
      for (size_t i = 0; i < fs.size(); ++i) {
        const auto x = x_min + (static_cast<double>(i) + 0.5) * dx;
        fs[i]        = f(x);
      }
      const auto i                = midpoint_rule(std::span(fs.data(), fs.size()), dx);

      const double expected_error = 5000.0 / static_cast<double>(Igor::sqr(Igor::sqr(N)));
      const double error          = std::abs(i - i_exp);
      if (error > expected_error) {
        Igor::Error("Incorrect result for Trapezoidal rule (N={}): expected error of <{:.6e} but "
                    "got {:.6e}",
                    N,
                    expected_error,
                    error);
        return 1;
      }
    }
  }
}
