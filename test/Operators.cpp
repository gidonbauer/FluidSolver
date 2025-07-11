#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "Operators.hpp"

using Float           = double;
constexpr Float X_MIN = 2.0;
constexpr Float X_MAX = 4.0;
constexpr Float Y_MIN = -1.0;
constexpr Float Y_MAX = 3.0;

constexpr Index NX = 200;
constexpr Index NY = 300;

constexpr auto DX = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

// -------------------------------------------------------------------------------------------------
auto test_eval_grid_at() noexcept -> bool {
  Igor::ScopeTimer timer("EvalGridAt");

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  // = Allocate memory =============================================================================

  // = Initialize grid =============================================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * DX;
  }
  for (Index i = 0; i < fs.xm.extent(0); ++i) {
    fs.xm[i] = (fs.x[i] + fs.x[i + 1]) / 2;
    fs.dx[i] = fs.x[i + 1] - fs.x[i];
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * DY;
  }
  for (Index j = 0; j < fs.ym.extent(0); ++j) {
    fs.ym[j] = (fs.y[j] + fs.y[j + 1]) / 2;
    fs.dy[j] = fs.y[j + 1] - fs.y[j];
  }
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  auto u = [](Float x, Float y) { return 12.0 * x + y; };
  auto v = [](Float x, Float y) { return x * y - 27.31415; };

  for (Index i = 0; i < fs.U.extent(0); ++i) {
    for (Index j = 0; j < fs.U.extent(1); ++j) {
      fs.U[i, j] = u(fs.x[i], fs.ym[j]);
    }
  }
  for (Index i = 0; i < fs.V.extent(0); ++i) {
    for (Index j = 0; j < fs.V.extent(1); ++j) {
      fs.V[i, j] = v(fs.xm[i], fs.y[j]);
    }
  }

  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);

  const Index N = 50;
  for (Index i = 0; i < N; ++i) {
    for (Index j = 0; j < N; ++j) {
      const auto x =
          (fs.xm[NX - 1] - fs.xm[0]) / static_cast<Float>(N - 1) * static_cast<Float>(i) + fs.xm[0];
      const auto y =
          (fs.ym[NY - 1] - fs.ym[0]) / static_cast<Float>(N - 1) * static_cast<Float>(j) + fs.ym[0];
      const auto [U, V] = eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, x, y);

      constexpr auto EPS = 100.0 * std::numeric_limits<Float>::epsilon();
      if (std::abs(U - u(x, y)) > EPS) {
        Igor::Warn("Incorrect interpolated velocity for U({:.4f}, {:.4f}), expected {:.6e} but got "
                   "{:.6e}: Error = {:.6e}",
                   x,
                   y,
                   u(x, y),
                   U,
                   std::abs(U - u(x, y)));
        return false;
      }

      if (std::abs(V - v(x, y)) > EPS) {
        Igor::Warn("Incorrect interpolated velocity for V({:.4f}, {:.4f}), expected {:.6e} but got "
                   "{:.6e}: Error = {:.6e}",
                   x,
                   y,
                   v(x, y),
                   V,
                   std::abs(V - v(x, y)));
        return false;
      }
    }
  }
  return true;
}

// -------------------------------------------------------------------------------------------------
auto test_gradient_centered_points() noexcept -> bool {
  Igor::ScopeTimer timer("GradientCenteredPoints");

  FS<Float, NX, NY> fs{};
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * DX;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * DY;
  }
  init_mid_and_delta(fs);

  Matrix<Float, NX, NY> f{};
  Matrix<Float, NX, NY> dfdx{};
  Matrix<Float, NX, NY> dfdy{};
  Matrix<Float, NX, NY> dfdxx{};
  Matrix<Float, NX, NY> dfdyy{};
  Matrix<Float, NX, NY> dfdxy{};
  Matrix<Float, NX, NY> dfdyx{};

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      const Float x = fs.xm[i] - 0.5;
      const Float y = fs.ym[j] - 0.5;
      f[i, j]       = x * x + x * y + y * y;
    }
  }

  calc_grad_of_centered_points(f, DX, DY, dfdx, dfdy);
  calc_grad_of_centered_points(dfdx, DX, DY, dfdxx, dfdxy);
  calc_grad_of_centered_points(dfdy, DX, DY, dfdyx, dfdyy);

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      const Float x              = fs.xm[i] - 0.5;
      const Float y              = fs.ym[j] - 0.5;
      const Float dfdx_expected  = 2.0 * x + y;
      const Float dfdy_expected  = x + 2.0 * y;
      const Float dfdxx_expected = 2.0;
      const Float dfdyy_expected = 2.0;
      const Float dfdxy_expected = 1.0;
      const Float dfdyx_expected = 1.0;

      constexpr Float EPS = 2e-10;

      if (std::abs(dfdx[i, j] - dfdx_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdx={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdx_expected,
            dfdx[i, j],
            std::abs(dfdx_expected - dfdx[i, j]));
        return false;
      }

      if (std::abs(dfdy[i, j] - dfdy_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdy={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdy_expected,
            dfdy[i, j],
            std::abs(dfdy_expected - dfdy[i, j]));
        return false;
      }

      if (std::abs(dfdxx[i, j] - dfdxx_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdxx={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdxx_expected,
            dfdxx[i, j],
            std::abs(dfdxx_expected - dfdxx[i, j]));
        return false;
      }

      if (std::abs(dfdyy[i, j] - dfdyy_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdyy={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdyy_expected,
            dfdyy[i, j],
            std::abs(dfdyy_expected - dfdyy[i, j]));
        return false;
      }

      if (std::abs(dfdxy[i, j] - dfdxy_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdxy={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdxy_expected,
            dfdxy[i, j],
            std::abs(dfdxy_expected - dfdxy[i, j]));
        return false;
      }

      if (std::abs(dfdyx[i, j] - dfdyx_expected) > EPS) {
        Igor::Warn(
            "Incorrect derivative at ({}, {}), expected dfdyx={:.6e} but is {:.6e}: error={:.6e}",
            fs.xm[i],
            fs.ym[j],
            dfdyx_expected,
            dfdyx[i, j],
            std::abs(dfdyx_expected - dfdyx[i, j]));
        return false;
      }
    }
  }

  return true;
}

auto main() -> int {
  bool success = true;
  success      = success && test_eval_grid_at();
  success      = success && test_gradient_centered_points();

  return success ? 0 : 1;
}
