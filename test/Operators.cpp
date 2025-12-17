#include <atomic>
#include <random>

#include <omp.h>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"

using Float            = double;
constexpr Float X_MIN  = 2.0;
constexpr Float X_MAX  = 4.0;
constexpr Float Y_MIN  = -1.0;
constexpr Float Y_MAX  = 3.0;

constexpr Index NX     = 200;
constexpr Index NY     = 300;
constexpr Index NGHOST = 1;

// -------------------------------------------------------------------------------------------------
auto test_eval_grid_at() noexcept -> bool {
  Igor::ScopeTimer timer("EvalGridAt");

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{};

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  // = Allocate memory =============================================================================

  // = Initialize grid =============================================================================
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  auto u = [](Float x, Float y) { return 12.0 * x + y; };
  auto v = [](Float x, Float y) { return x * y - 27.31415; };

  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      fs.curr.U(i, j) = u(fs.x(i), fs.ym(j));
    }
  }
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      fs.curr.V(i, j) = v(fs.xm(i), fs.y(j));
    }
  }

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);

  const Index N = 50;
  for (Index i = 0; i < N; ++i) {
    for (Index j = 0; j < N; ++j) {
      const auto x =
          (fs.xm(NX - 1) - fs.xm(0)) / static_cast<Float>(N - 1) * static_cast<Float>(i) + fs.xm(0);
      const auto y =
          (fs.ym(NY - 1) - fs.ym(0)) / static_cast<Float>(N - 1) * static_cast<Float>(j) + fs.ym(0);
      const auto [U, V]  = eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, x, y);

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

  FS<Float, NX, NY, NGHOST + 4> fs{};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  Field2D<Float, NX, NY, NGHOST + 4> f{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdx{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdy{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdxx{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdyy{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdxy{};
  Field2D<Float, NX, NY, NGHOST + 4> dfdyx{};

  for_each_a<Exec::Parallel>(f, [&](Index i, Index j) {
    const Float x = fs.xm(i) - 0.5;
    const Float y = fs.ym(j) - 0.5;
    f(i, j)       = x * x + x * y + y * y;
  });

  calc_grad_of_centered_points(f, fs.dx, fs.dy, dfdx, dfdy);
  calc_grad_of_centered_points(dfdx, fs.dx, fs.dy, dfdxx, dfdxy);
  calc_grad_of_centered_points(dfdy, fs.dx, fs.dy, dfdyx, dfdyy);

  bool all_success = true;
  for_each_a(f, [&](Index i, Index j) {
    const Float x              = fs.xm(i) - 0.5;
    const Float y              = fs.ym(j) - 0.5;
    const Float dfdx_expected  = 2.0 * x + y;
    const Float dfdy_expected  = x + 2.0 * y;
    const Float dfdxx_expected = 2.0;
    const Float dfdyy_expected = 2.0;
    const Float dfdxy_expected = 1.0;
    const Float dfdyx_expected = 1.0;

    constexpr Float EPS        = 2e-10;

    if (std::abs(dfdx(i, j) - dfdx_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdx={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdx_expected,
                 dfdx(i, j),
                 std::abs(dfdx_expected - dfdx(i, j)));
      all_success = false;
    }

    if (std::abs(dfdy(i, j) - dfdy_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdy={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdy_expected,
                 dfdy(i, j),
                 std::abs(dfdy_expected - dfdy(i, j)));
      all_success = false;
    }

    if (std::abs(dfdxx(i, j) - dfdxx_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdxx={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdxx_expected,
                 dfdxx(i, j),
                 std::abs(dfdxx_expected - dfdxx(i, j)));
      all_success = false;
    }

    if (std::abs(dfdyy(i, j) - dfdyy_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdyy={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdyy_expected,
                 dfdyy(i, j),
                 std::abs(dfdyy_expected - dfdyy(i, j)));
      all_success = false;
    }

    if (std::abs(dfdxy(i, j) - dfdxy_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdxy={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdxy_expected,
                 dfdxy(i, j),
                 std::abs(dfdxy_expected - dfdxy(i, j)));
      all_success = false;
    }

    if (std::abs(dfdyx(i, j) - dfdyx_expected) > EPS) {
      Igor::Warn("{}, {}: Incorrect derivative at ({}, {}), expected dfdyx={:.6e} but is {:.6e}: "
                 "error={:.6e}",
                 i,
                 j,
                 fs.xm(i),
                 fs.ym(j),
                 dfdyx_expected,
                 dfdyx(i, j),
                 std::abs(dfdyx_expected - dfdyx(i, j)));
      all_success = false;
    }
  });

  return all_success;
}

// -------------------------------------------------------------------------------------------------
auto test_staggered_integral() -> bool {
  Igor::ScopeTimer timer("StaggeredIntegral");
  bool res = true;

  FS<Float, NX, NY, NGHOST> fs{};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  auto rho  = [](Float x, Float y) { return 12.0 * x + x * y * y; };
  auto rhoU = [](Float x, Float y) { return (12.0 * x + x * y * y) * std::sin(x + y); };
  auto rhoV = [](Float x, Float y) { return (12.0 * x + x * y * y) * std::cos(x + y); };

  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      fs.curr.rho_u_stag(i, j) = rho(fs.x(i), fs.ym(j));
      fs.curr.U(i, j)          = rhoU(fs.x(i), fs.ym(j)) / rho(fs.x(i), fs.ym(j));
    }
  }
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      fs.curr.rho_v_stag(i, j) = rho(fs.xm(i), fs.y(j));
      fs.curr.V(i, j)          = rhoV(fs.xm(i), fs.y(j)) / rho(fs.xm(i), fs.y(j));
    }
  }

  Float mass_expected       = 0.0;
  Float momentum_x_expected = 0.0;
  Float momentum_y_expected = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      mass_expected       += quadrature<64>(rho, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1));
      momentum_x_expected += quadrature<64>(rhoU, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1));
      momentum_y_expected += quadrature<64>(rhoV, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1));
    }
  }

  Float mass       = 0.0;
  Float momentum_x = 0.0;
  Float momentum_y = 0.0;
  calc_conserved_quantities(fs, mass, momentum_x, momentum_y);

  const auto mass_error       = std::abs(mass - mass_expected);
  const auto momentum_x_error = std::abs(momentum_x - momentum_x_expected);
  const auto momentum_y_error = std::abs(momentum_y - momentum_y_expected);

  const Float TOL             = 10.0 * Igor::sqr(std::min(fs.dx, fs.dy));
  if (mass_error > TOL) {
    Igor::Warn("Calculated incorrect mass: expected {:.6e} but got {:.6e} => error = {:.6e}",
               mass_expected,
               mass,
               mass_error);
    res = false;
  }
  if (momentum_x_error > TOL) {
    Igor::Warn("Calculated incorrect momentum in x-direction: expected {:.6e} but got {:.6e} => "
               "error = {:.6e}",
               momentum_x_expected,
               momentum_x,
               momentum_x_error);
    res = false;
  }
  if (momentum_y_error > TOL) {
    Igor::Warn("Calculated incorrect momentum in y-direction: expected {:.6e} but got {:.6e} => "
               "error = {:.6e}",
               momentum_y_expected,
               momentum_y,
               momentum_y_error);
    res = false;
  }

  return res;
}

auto test_atomics() -> bool {
  Igor::ScopeTimer timer("Atomics");

  Field2D<Float, 2048, 2048, 0> field;
  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<Float> dist(-10.0, 10.0);
  std::generate_n(field.get_data(), field.size(), [&]() { return dist(rng); });

  const auto expected_int         = integrate(1.0, 1.0, field);

  std::atomic<Float> parallel_int = 0.0;
  for_each_i<Exec::Parallel>(field, [&](Index i, Index j) { parallel_int += field(i, j); });

  return std::abs(expected_int - parallel_int) < 1e-8;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  omp_set_num_threads(4);

  bool success      = true;
  bool test_success = true;

  test_success      = test_eval_grid_at();
  if (!test_success) { Igor::Warn("EvalGridAt failed."); }
  success      = success && test_success;

  test_success = test_gradient_centered_points();
  if (!test_success) { Igor::Warn("GradientCenteredPoints failed."); }
  success      = success && test_success;

  test_success = test_staggered_integral();
  if (!test_success) { Igor::Warn("StaggeredIntegral failed."); }
  success      = success && test_success;

  test_success = test_atomics();
  if (!test_success) { Igor::Warn("Atomics failed."); }
  success = success && test_success;

  return success ? 0 : 1;
}
