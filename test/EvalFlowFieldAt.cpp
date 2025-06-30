#include "Container.hpp"
#include "Operators.hpp"

using Float           = double;
constexpr Float X_MIN = 2.0;
constexpr Float X_MAX = 4.0;
constexpr Float Y_MIN = -1.0;
constexpr Float Y_MAX = 3.0;

constexpr Index NX = 100;
constexpr Index NY = 200;

constexpr auto DX = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

auto main() -> int {
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
        return 1;
      }

      if (std::abs(V - v(x, y)) > EPS) {
        Igor::Warn("Incorrect interpolated velocity for V({:.4f}, {:.4f}), expected {:.6e} but got "
                   "{:.6e}: Error = {:.6e}",
                   x,
                   y,
                   v(x, y),
                   V,
                   std::abs(V - v(x, y)));
        return 1;
      }
    }
  }
}
