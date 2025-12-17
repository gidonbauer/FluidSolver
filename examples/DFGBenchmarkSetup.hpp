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
enum : size_t { X, Y };

constexpr Index NY           = 128;
constexpr Index NX           = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST       = 1;

constexpr Float T_END        = 8.0;
constexpr Float DT_MAX       = 1e-2;
constexpr Float CFL_MAX      = 0.5;
constexpr Float DT_WRITE     = 2e-2;

constexpr Float VISC         = 1e-3;
constexpr Float RHO          = 1.0;

constexpr Float CX           = 0.2;
constexpr Float CY           = 0.2;
constexpr Float R0           = 0.05;
constexpr Float L            = 2.0 * R0;
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};
constexpr auto normal_immersed_wall = [](Float x, Float y) -> std::array<Float, 2> {
  const auto d = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));
  return {
      (x - CX) / d,
      (y - CY) / d,
  };
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

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

#if 1
template <typename Float_, Index NX_, Index NY_, Index NGHOST_>
constexpr auto eval_field_at(const Vector<Float_, NX_, NGHOST_>& xm,
                             const Vector<Float_, NY_, NGHOST_>& ym,
                             const Matrix<Float_, NX_, NY_, NGHOST_>& field,
                             Float_ x,
                             Float_ y) -> Float_ {
  const auto dx    = xm(1) - xm(0);
  const auto dy    = ym(1) - ym(0);

  auto get_indices = []<Index N>(Float pos,
                                 const Vector<Float, N, NGHOST>& grid,
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
constexpr auto eval_field_at(const Vector<Float_, NX_, NGHOST_>& xm,
                             const Vector<Float_, NY_, NGHOST_>& ym,
                             const Matrix<Float_, NX_, NY_, NGHOST_>& field,
                             Float_ x,
                             Float_ y) -> Float_ {
  return bilinear_interpolate(xm, ym, field, x, y);
}
#endif

// -------------------------------------------------------------------------------------------------
constexpr auto calc_p_diff(const FS<Float, NX, NY, NGHOST>& fs) -> Float {
  constexpr std::array<Float, 2> a1 = {0.15, 0.2};
  constexpr std::array<Float, 2> a2 = {0.25, 0.2};
  static_assert(a1[Y] == a2[Y]);

#if 1
  const auto p1 = eval_field_at(fs.xm, fs.ym, fs.p, a1[X], a1[Y]);
  const auto p2 = eval_field_at(fs.xm, fs.ym, fs.p, a2[X], a2[Y]);
#else
  const auto jprev = static_cast<Index>(std::floor((a1[Y] - fs.ym(0)) / fs.dy));
  const auto jnext = jprev + 1;
  IGOR_ASSERT(fs.ym(jprev) <= a1[Y] && a1[Y] <= fs.ym(jnext),
              "Incorrect indices: {:.6e} <= {:.6e} <= {:.6e}",
              fs.ym(jprev),
              a1[Y],
              fs.ym(jnext));

  const auto i1 = static_cast<Index>(std::floor((a1[X] - fs.xm(0)) / fs.dx));
  const auto p1 =
      (fs.p(i1, jnext) - fs.p(i1, jprev)) / fs.dy * (a1[Y] - fs.ym(jprev)) + fs.p(i1, jprev);

  const auto i2 = static_cast<Index>(std::floor((a2[X] - fs.xm(0)) / fs.dx + 1.0));
  const auto p2 =
      (fs.p(i2, jnext) - fs.p(i2, jprev)) / fs.dy * (a1[Y] - fs.ym(jprev)) + fs.p(i2, jprev);
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
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = CX + normal_x * R0;
    const auto y        = CY + normal_y * R0;

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
}

constexpr auto calc_C_D(const FS<Float, NX, NY, NGHOST>& fs, Float t) {
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = CX + normal_x * R0;
    const auto y        = CY + normal_y * R0;

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
}

#endif  // DFG_BENCHMARK_SETUP_HPP_
