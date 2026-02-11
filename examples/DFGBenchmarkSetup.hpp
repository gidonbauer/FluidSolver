#ifndef DFG_BENCHMARK_SETUP_HPP_
#define DFG_BENCHMARK_SETUP_HPP_

#define DFG_BENCHMARK 1
// #define SYMMETRIC

#if !(DFG_BENCHMARK == 1 || DFG_BENCHMARK == 2 || DFG_BENCHMARK == 3)
#error "DFG_BENCHMARK must be 1, 2, or 3"
#endif

#include <numbers>

#include <Igor/Math.hpp>

#include "BoundaryConditions.hpp"
#include "Container.hpp"
#include "FS.hpp"
#include "Geometry.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"

// = Config ========================================================================================
using Float           = double;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 2.2;
constexpr Float Y_MIN = 0.0;
#ifdef SYMMETRIC
constexpr Float Y_MAX = 0.4;
#else
constexpr Float Y_MAX = 0.41;
#endif  // SYMMETRIC

constexpr Index NY       = 256;
constexpr Index NX       = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST   = 1;

constexpr Float T_END    = 8.0;
constexpr Float DT_MAX   = 1e-2;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = 2e-2;

constexpr Float VISC     = 1e-3;
constexpr Float RHO      = 1.0;

constexpr Circle wall{.x = 0.2, .y = 0.2, .r = 0.05};
constexpr Float L            = 2.0 * wall.r;
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(wall.contains({x, y}));
};
constexpr auto normal_immersed_wall = [](Float x, Float y) -> Vector2<Float> {
  const auto d = std::sqrt(Igor::sqr(x - wall.x) + Igor::sqr(y - wall.y));
  return {.x = (x - wall.x) / d, .y = (y - wall.y) / d};
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, [[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
  constexpr Float height = Y_MAX - Y_MIN;
#if DFG_BENCHMARK == 1
  const Float U = 0.3;
#elif DFG_BENCHMARK == 2
  const Float U = 1.5;
#else
  const auto U = 1.5 * std::sin(std::numbers::pi * t / 8.0);
#endif
  return (4.0 * U * y * (height - y)) / Igor::sqr(height);
}

constexpr auto U_mean([[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
#if DFG_BENCHMARK == 1
  const Float U = 0.3;
#elif DFG_BENCHMARK == 2
  const Float U = 1.5;
#else
  const auto U = 1.5 * std::sin(std::numbers::pi * t / 8.0);
#endif
  return 2.0 / 3.0 * U;
}

static_assert(Igor::abs(U_mean(0.0) -
                        quadrature([](Float y) { return U_in(y, 0.0); }, Y_MIN, Y_MAX) /
                            (Y_MAX - Y_MIN)) < 1e-8,
              "U_mean does not match U_in.");
static_assert(Igor::abs(U_mean(4.0) -
                        quadrature([](Float y) { return U_in(y, 4.0); }, Y_MIN, Y_MAX) /
                            (Y_MAX - Y_MIN)) < 1e-8,
              "U_mean does not match U_in.");
static_assert(Igor::abs(U_mean(8.0) -
                        quadrature([](Float y) { return U_in(y, 8.0); }, Y_MIN, Y_MAX) /
                            (Y_MAX - Y_MIN)) < 1e-8,
              "U_mean does not match U_in.");

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

#if 0
template <typename Float_, Index NX_, Index NY_, Index NGHOST_>
constexpr auto eval_field_at(const Field1D<Float_, NX_, NGHOST_>& xm,
                             const Field1D<Float_, NY_, NGHOST_>& ym,
                             const Field2D<Float_, NX_, NY_, NGHOST_>& field,
                             Float_ x,
                             Float_ y) -> Float_ {
  const auto dx    = xm(1) - xm(0);
  const auto dy    = ym(1) - ym(0);

  auto get_indices = []<Index N>(Float pos,
                                 const Field1D<Float, N, NGHOST>& grid,
                                 Float delta) -> std::pair<Index, Index> {
    if (pos <= grid(0)) { return {0, 0}; }
    if (pos >= grid(N - 1)) { return {N - 1, N - 1}; }
    const auto prev = static_cast<Index>(std::floor((pos - grid(0)) / delta));
    const auto next = static_cast<Index>(std::floor((pos - grid(0)) / delta + 1.0));
    return {prev, next};
  };

  const auto i_idxs = get_indices(x, xm, dx);
  const auto j_idxs = get_indices(y, ym, dy);

  const std::array<std::pair<Index, Index>, 4> idx_pairs{
      std::pair<Index, Index>{i_idxs.first, j_idxs.first},
      std::pair<Index, Index>{i_idxs.first, j_idxs.second},
      std::pair<Index, Index>{i_idxs.second, j_idxs.first},
      std::pair<Index, Index>{i_idxs.second, j_idxs.second},
  };
  Float min_dist  = std::numeric_limits<Float>::max();
  size_t pair_idx = idx_pairs.size();
  for (size_t pi = 0; pi < idx_pairs.size(); ++pi) {
    const auto px    = xm(idx_pairs[pi].first);
    const auto py    = ym(idx_pairs[pi].second);
    const Float dist = Igor::sqr(x - px) + Igor::sqr(y - py);
    if (dist < min_dist && immersed_wall(px, py) < 1.0) {
      min_dist = dist;
      pair_idx = pi;
    }
  }

  IGOR_ASSERT(pair_idx < idx_pairs.size(),
              "Did not find any suitable field values outside of the immersed wall.");
  return field(idx_pairs[pair_idx].first, idx_pairs[pair_idx].second);
}
#else
template <typename Float_, Index NX_, Index NY_, Index NGHOST_>
constexpr auto eval_field_at(const Field1D<Float_, NX_, NGHOST_>& xm,
                             const Field1D<Float_, NY_, NGHOST_>& ym,
                             const Field2D<Float_, NX_, NY_, NGHOST_>& field,
                             Float_ x,
                             Float_ y) -> Float_ {
  return bilinear_interpolate(xm, ym, field, x, y);
}
#endif

// -------------------------------------------------------------------------------------------------
constexpr auto calc_p_diff(const FS<Float, NX, NY, NGHOST>& fs) -> Float {
  constexpr Vector2<Float> a1 = {.x = 0.15, .y = 0.2};
  constexpr Vector2<Float> a2 = {.x = 0.25, .y = 0.2};
  static_assert(a1.y == a2.y);

#if 1
  const auto p1 = eval_field_at(fs.xm, fs.ym, fs.p, a1.x, a1.y);
  const auto p2 = eval_field_at(fs.xm, fs.ym, fs.p, a2.x, a2.y);
#else
  const auto jprev = static_cast<Index>(std::floor((a1.y - fs.ym(0)) / fs.dy));
  const auto jnext = jprev + 1;
  IGOR_ASSERT(fs.ym(jprev) <= a1.y && a1.y <= fs.ym(jnext),
              "Incorrect indices: {:.6e} <= {:.6e} <= {:.6e}",
              fs.ym(jprev),
              a1.y,
              fs.ym(jnext));

  const auto i1 = static_cast<Index>(std::floor((a1.x - fs.xm(0)) / fs.dx));
  const auto p1 =
      (fs.p(i1, jnext) - fs.p(i1, jprev)) / fs.dy * (a1.y - fs.ym(jprev)) + fs.p(i1, jprev);

  const auto i2 = static_cast<Index>(std::floor((a2.x - fs.xm(0)) / fs.dx + 1.0));
  const auto p2 =
      (fs.p(i2, jnext) - fs.p(i2, jprev)) / fs.dy * (a1.y - fs.ym(jprev)) + fs.p(i2, jprev);
#endif

  return p1 - p2;
}

// -------------------------------------------------------------------------------------------------
constexpr auto calc_Re(Float t) -> Float { return RHO * U_mean(t) * L / VISC; }

// -------------------------------------------------------------------------------------------------
constexpr auto calc_coefficient(auto&& f, const FS<Float, NX, NY, NGHOST>& fs, Float t) -> Float {
#if 1
  const auto force = quadrature<64>(f, 0.0, std::numbers::pi) +
                     quadrature<64>(f, std::numbers::pi, 2.0 * std::numbers::pi);
#else
  const auto force = quadrature<16>(f, 0.0, 2.0 * std::numbers::pi);
#endif

  return (2.0 * force) / (fs.rho_gas * Igor::sqr(U_mean(t)) * L);
}

constexpr auto calc_C_L(const FS<Float, NX, NY, NGHOST>& fs, Float t) {
#if 0
  static Field2D<Float, NX, NY, NGHOST> dudx{};
  static Field2D<Float, NX, NY, NGHOST> dvdy{};
  static Field2D<Float, NX + 1, NY + 1, NGHOST> dudy{};
  static Field2D<Float, NX + 1, NY + 1, NGHOST> dvdx{};

  for_each_a<Exec::Parallel>(dudx, [&](Index i, Index j) {
    dudx(i, j) = (fs.curr.U(i + 1, j) - fs.curr.U(i, j)) / fs.dx;
  });
  for_each_a<Exec::Parallel>(dvdy, [&](Index i, Index j) {
    dvdy(i, j) = (fs.curr.V(i, j + 1) - fs.curr.V(i, j)) / fs.dy;
  });
  for_each_i<Exec::Parallel>(dudy, [&](Index i, Index j) {
    dudy(i, j) = (fs.curr.U(i, j) - fs.curr.U(i, j - 1)) / fs.dy;
  });
  for_each_i<Exec::Parallel>(dvdx, [&](Index i, Index j) {
    dvdx(i, j) = (fs.curr.V(i, j) - fs.curr.V(i - 1, j)) / fs.dx;
  });

  auto f = [&](Float theta) {
    const auto normal_x    = std::cos(theta);
    const auto normal_y    = std::sin(theta);
    const auto x           = wall.x + normal_x * wall.r;
    const auto y           = wall.y + normal_y * wall.r;

    const auto p_interp    = eval_field_at(fs.xm, fs.ym, fs.p, x, y);
    const auto dvdy_interp = eval_field_at(fs.xm, fs.ym, dvdy, x, y);
    const auto dudy_interp = eval_field_at(fs.x, fs.y, dudy, x, y);
    const auto dvdx_interp = eval_field_at(fs.x, fs.y, dvdx, x, y);

    return -(fs.visc_gas * (dudy_interp + dvdx_interp)) * normal_x +
           (p_interp + 2.0 * fs.visc_gas * dvdy_interp) * normal_y;
  };

  return calc_coefficient(f, fs, t);
#elif 1
  Float lift_force = 0.0;
  for_each_i(fs.xm, [&](Index i) {
    const Float x = fs.xm(i);
    if (x < wall.x - wall.r || x > wall.x + wall.r) { return; }

    // (x - cx)^2 + (y - cy)^2 = r^2
    // y_left = -sqrt(r^2 - (x - cx)^2) + cy
    // y_right = sqrt(r^2 - (x - cx)^2) + cy

    const Float y_bottom    = -std::sqrt(Igor::sqr(wall.r) - Igor::sqr(x - wall.x)) + wall.y;
    const Float y_top       = std::sqrt(Igor::sqr(wall.r) - Igor::sqr(x - wall.x)) + wall.y;

    const auto j_bottom     = static_cast<Index>(std::floor((y_bottom - fs.y(0)) / fs.dy));
    const auto j_top        = static_cast<Index>(std::floor((y_top - fs.y(0)) / fs.dy));

    const Float p_bottom    = fs.p(i, j_bottom);
    const Float p_top       = fs.p(i, j_top);

    const Float dvdy_bottom = (fs.curr.V(i, j_bottom) - fs.curr.V(i, j_bottom - 1)) / fs.dy;
    const Float dvdy_top    = (fs.curr.V(i, j_top + 1) - fs.curr.V(i, j_top)) / fs.dy;

    lift_force += -((p_bottom - p_top) + 2.0 * fs.visc_gas * (dvdy_top - dvdy_bottom)) * fs.dx;
  });

  return (2.0 * lift_force) / (fs.rho_gas * Igor::sqr(U_mean(t)) * L);
#else
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = wall.x + normal_x * wall.r;
    const auto y        = wall.y + normal_y * wall.r;

    const auto p        = eval_field_at(fs.xm, fs.ym, fs.p, x, y);

    const auto delta    = std::min(fs.dx, fs.dy) / 2.0;
    auto calc_Ut        = [&](Float local_x, Float local_y) {
      const auto u = eval_field_at(fs.x, fs.ym, fs.curr.U, local_x, local_y);
      const auto v = eval_field_at(fs.xm, fs.y, fs.curr.V, local_x, local_y);
      return u * normal_y - v * normal_x;
    };
    const auto Ut_1 = calc_Ut(x, y);
    const auto Ut_2 = calc_Ut(x + delta * normal_x, y + delta * normal_y);
    const auto dudn = (Ut_2 - Ut_1) / delta;

    return -p * normal_y - fs.rho_gas * fs.visc_gas * dudn * normal_x;
  };

  return calc_coefficient(f, fs, t);
#endif
}

constexpr auto calc_C_D(const FS<Float, NX, NY, NGHOST>& fs, Float t) -> Float {
#if 0
  static Field2D<Float, NX, NY, NGHOST> dudx{};
  static Field2D<Float, NX, NY, NGHOST> dvdy{};
  static Field2D<Float, NX + 1, NY + 1, NGHOST> dudy{};
  static Field2D<Float, NX + 1, NY + 1, NGHOST> dvdx{};

  for_each_a<Exec::Parallel>(dudx, [&](Index i, Index j) {
    dudx(i, j) = (fs.curr.U(i + 1, j) - fs.curr.U(i, j)) / fs.dx;
  });
  for_each_a<Exec::Parallel>(dvdy, [&](Index i, Index j) {
    dvdy(i, j) = (fs.curr.V(i, j + 1) - fs.curr.V(i, j)) / fs.dy;
  });
  for_each_i<Exec::Parallel>(dudy, [&](Index i, Index j) {
    dudy(i, j) = (fs.curr.U(i, j) - fs.curr.U(i, j - 1)) / fs.dy;
  });
  for_each_i<Exec::Parallel>(dvdx, [&](Index i, Index j) {
    dvdx(i, j) = (fs.curr.V(i, j) - fs.curr.V(i - 1, j)) / fs.dx;
  });

  auto f = [&](Float theta) {
    const auto normal_x    = std::cos(theta);
    const auto normal_y    = std::sin(theta);
    const auto x           = wall.x + normal_x * wall.r;
    const auto y           = wall.y + normal_y * wall.r;

    const auto p_interp    = eval_field_at(fs.xm, fs.ym, fs.p, x, y);
    const auto dudx_interp = eval_field_at(fs.xm, fs.ym, dudx, x, y);
    const auto dudy_interp = eval_field_at(fs.x, fs.y, dudy, x, y);
    const auto dvdx_interp = eval_field_at(fs.x, fs.y, dvdx, x, y);

    return (-p_interp + 2.0 * fs.visc_gas * dudx_interp) * normal_x +
           (fs.visc_gas * (dudy_interp + dvdx_interp)) * normal_y;
  };

  return calc_coefficient(f, fs, t);
#elif 1
  Float drag_force = 0.0;

  for_each_i(fs.ym, [&](Index j) {
    const Float y = fs.ym(j);
    if (y < wall.y - wall.r || y > wall.y + wall.r) { return; }

    // (x - cx)^2 + (y - cy)^2 = r^2
    // x_left = -sqrt(r^2 - (y - cy)^2) + cx
    // x_right = sqrt(r^2 - (y - cy)^2) + cx

    const Float x_left     = -std::sqrt(Igor::sqr(wall.r) - Igor::sqr(y - wall.y)) + wall.x;
    const Float x_right    = std::sqrt(Igor::sqr(wall.r) - Igor::sqr(y - wall.y)) + wall.x;

    const auto i_left      = static_cast<Index>(std::floor((x_left - fs.x(0)) / fs.dx));
    const auto i_right     = static_cast<Index>(std::floor((x_right - fs.x(0)) / fs.dx));

    const Float p_left     = fs.p(i_left, j);
    const Float p_right    = fs.p(i_right, j);

    const Float dudx_left  = (fs.curr.U(i_left, j) - fs.curr.U(i_left - 1, j)) / fs.dx;
    const Float dudx_right = (fs.curr.U(i_right + 1, j) - fs.curr.U(i_right, j)) / fs.dx;

    drag_force += ((p_left - p_right) + 2.0 * fs.visc_gas * (dudx_right - dudx_left)) * fs.dy;
  });

  // for_each_i(fs.xm, [&](Index i) {
  //   const Float x = fs.xm(i);
  //   if (x < wall.x - wall.r || x > wall.x + wall.r) { return; }

  //   // (x - cx)^2 + (y - cy)^2 = r^2
  //   // y_left = -sqrt(r^2 - (x - cx)^2) + cy
  //   // y_right = sqrt(r^2 - (x - cx)^2) + cy

  //   const Float y_bottom = -std::sqrt(Igor::sqr(wall.r) - Igor::sqr(x - wall.x)) + wall.y;
  //   const Float y_top    = std::sqrt(Igor::sqr(wall.r) - Igor::sqr(x - wall.x)) + wall.y;

  //   const auto j_bottom  = static_cast<Index>(std::floor((y_bottom - fs.y(0)) / fs.dy));
  //   const auto j_top     = static_cast<Index>(std::floor((y_top - fs.y(0)) / fs.dy));

  //   const Float dvdx_bottom =
  //       (fs.curr.V(i + 1, j_bottom) - fs.curr.V(i - 1, j_bottom)) / (2.0 * fs.dx);
  //   const Float dvdx_top    = (fs.curr.V(i + 1, j_top) - fs.curr.V(i - 1, j_top)) / (2.0 *
  //   fs.dx);

  //   const Float dudy_bottom = (fs.curr.U(i, j_bottom) - fs.curr.U(i, j_bottom - 1)) / fs.dy;
  //   const Float dudy_top    = (fs.curr.U(i, j_top + 1) - fs.curr.U(i, j_top)) / fs.dy;

  //   drag_force += fs.visc_gas * ((dvdx_bottom - dvdx_top) + (dudy_top - dudy_bottom)) * fs.dx;
  // });

  return (2.0 * drag_force) / (fs.rho_gas * Igor::sqr(U_mean(t)) * L);
#else
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = wall.x + normal_x * wall.r;
    const auto y        = wall.y + normal_y * wall.r;

    const auto p        = eval_field_at(fs.xm, fs.ym, fs.p, x, y);

#if 0
    return -p * normal_x;
#else
    const auto delta = std::min(fs.dx, fs.dy) / 2.0;
    auto calc_Ut     = [&](Float local_x, Float local_y) {
      const auto u = eval_field_at(fs.x, fs.ym, fs.curr.U, local_x, local_y);
      const auto v = eval_field_at(fs.xm, fs.y, fs.curr.V, local_x, local_y);
      return u * normal_y - v * normal_x;
    };
    const auto Ut_1 = calc_Ut(x, y);
    const auto Ut_2 = calc_Ut(x + delta * normal_x, y + delta * normal_y);
    const auto dudn = (Ut_2 - Ut_1) / delta;

    return -p * normal_x + fs.rho_gas * fs.visc_gas * dudn * normal_y;
#endif
  };

  return calc_coefficient(f, fs, t);
#endif
}

#endif  // DFG_BENCHMARK_SETUP_HPP_
