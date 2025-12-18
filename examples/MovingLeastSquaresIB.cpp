#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float            = double;

constexpr Index NX     = 32;
constexpr Index NY     = 32;
constexpr Index NGHOST = 1;

constexpr Float X_MIN  = 0.0;
constexpr Float X_MAX  = 2.0 * std::numbers::pi_v<Float>;
constexpr Float Y_MIN  = 0.0;
constexpr Float Y_MAX  = 2.0 * std::numbers::pi_v<Float>;

constexpr Float VISC   = 1e-3;
constexpr Float RHO    = 1.0;
// = Config ========================================================================================

// = Analytical solution ===========================================================================
auto F(Float t) -> Float { return std::exp(-2.0 * VISC / RHO * t); }
auto u_analytical(Float x, Float y, Float t) -> Float { return std::sin(x) * std::cos(y) * F(t); }
auto v_analytical(Float x, Float y, Float t) -> Float { return -std::cos(x) * std::sin(y) * F(t); }
// = Analytical solution ===========================================================================

// =================================================================================================
struct Vector2 {
  Float x, y;
};

struct IndexPair {
  Index first, second;
};

// =================================================================================================
constexpr auto dist(const Vector2& v1, const Vector2& v2) noexcept -> Float {
  return std::sqrt(Igor::sqr(v1.x - v2.x) + Igor::sqr(v1.y - v2.y));
}

constexpr auto weighted_dist(const Vector2& v1, const Vector2& v2, const Vector2 w) noexcept
    -> Float {
  return std::sqrt(Igor::sqr((v1.x - v2.x) / w.x) + Igor::sqr((v1.y - v2.y) / w.y));
}

// =================================================================================================
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto find_nearest_neighbor(const Field1D<Float, NX, NGHOST>& xm,
                                     const Field1D<Float, NY, NGHOST>& ym,
                                     Float x,
                                     Float y) -> IndexPair {
  const auto dx = xm(1) - xm(0);
  const auto dy = ym(1) - ym(0);

  auto get_indices =
      []<Index N>(Float pos, const Field1D<Float, N, NGHOST>& grid, Float delta) -> IndexPair {
    if (pos <= grid(0)) { return {.first = 0, .second = 0}; }
    if (pos >= grid(N - 1)) { return {.first = N - 1, .second = N - 1}; }
    const auto prev = static_cast<Index>(std::floor((pos - grid(0)) / delta));
    const auto next = static_cast<Index>(std::floor((pos - grid(0)) / delta + 1.0));
    return {.first = prev, .second = next};
  };

  const auto i_idxs = get_indices(x, xm, dx);
  const auto j_idxs = get_indices(y, ym, dy);

  const std::array<IndexPair, 4> idx_pairs{
      IndexPair{i_idxs.first, j_idxs.first},
      IndexPair{i_idxs.first, j_idxs.second},
      IndexPair{i_idxs.second, j_idxs.first},
      IndexPair{i_idxs.second, j_idxs.second},
  };
  Float min_dist  = std::numeric_limits<Float>::max();
  size_t pair_idx = idx_pairs.size();
  for (size_t pi = 0; pi < idx_pairs.size(); ++pi) {
    const auto px    = xm(idx_pairs[pi].first);
    const auto py    = ym(idx_pairs[pi].second);
    const Float dist = Igor::sqr(x - px) + Igor::sqr(y - py);
    if (dist < min_dist) {
      min_dist = dist;
      pair_idx = pi;
    }
  }

  IGOR_ASSERT(pair_idx < idx_pairs.size(),
              "Did not find any suitable field values outside of the immersed wall.");
  return idx_pairs[pair_idx];
}

// =================================================================================================
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto eval_field_at_nn(const Field1D<Float, NX, NGHOST>& xm,
                                const Field1D<Float, NY, NGHOST>& ym,
                                const Field2D<Float, NX, NY, NGHOST>& field,
                                Float x,
                                Float y) -> Float {
  const auto [i, j] = find_nearest_neighbor(xm, ym, x, y);
  return field(i, j);
}

// = Shape function ================================================================================
constexpr auto weight_function(Vector2 pos, Vector2 center, Vector2 box_size) -> Float {
  const Float r = weighted_dist(pos, center, box_size);
  if (r <= 0.5) {
    return 2.0 / 3.0 - 4.0 * r * r + 4.0 * r * r * r;
  } else if (r <= 1.0) {
    return 4.0 / 3.0 - 4.0 * r + 4.0 * r * r - 4.0 / 3.0 * r * r * r;
  } else {
    return 0.0;
  }
}

template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto eval_field_at_sf(const Field1D<Float, NX, NGHOST>& xm,
                                const Field1D<Float, NY, NGHOST>& ym,
                                const Field2D<Float, NX, NY, NGHOST>& field,
                                Float x,
                                Float y) -> Float {
  constexpr Index NE = 5;

  const Float dx     = xm(1) - xm(0);
  const Float dy     = ym(1) - ym(0);
  const Vector2 box_size{
      .x = 1.2 * dx,
      .y = 1.2 * dy,
  };

  const Vector2 center = {.x = x, .y = y};
  Matrix<Float, 3, 3> A{};
  Matrix<Float, 3, NE> B{};
  Vector<Float, NE> U{};

  const auto [inn, jnn]                      = find_nearest_neighbor(xm, ym, x, y);
  const std::array<IndexPair, NE> neighbours = {
      IndexPair{inn, jnn},
      IndexPair{inn - 1, jnn},
      IndexPair{inn + 1, jnn},
      IndexPair{inn, jnn - 1},
      IndexPair{inn, jnn + 1},
  };
  for (size_t k = 0; k < neighbours.size(); ++k) {
    const auto [i, j]        = neighbours[k];
    const Vector2 pos        = {.x = xm(i), .y = ym(j)};
    const Vector<Float, 3> p = {1.0, pos.x, pos.y};
    const Float w            = weight_function(pos, center, box_size);

    for (Index ii = 0; ii < 3; ++ii) {
      for (Index jj = 0; jj < 3; ++jj) {
        A(ii, jj) += w * p(ii) * p(jj);
      }
    }
    for (Index ii = 0; ii < 3; ++ii) {
      B(ii, static_cast<Index>(k)) = w * p(ii);
    }
    U(static_cast<Index>(k)) = field(i, j);
  }

  Vector<Float, 3> Bu{};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < NE; ++j) {
      Bu(i) += B(i, j) * U(j);
    }
  }

  // Vector<Float, 3> a{};
  // solve_linear_system(A, Bu, a);
  // return a(0) * 1.0 + a(1) * x + a(2) * y;

  Vector<Float, 3> p = {1.0, x, y};
  Vector<Float, 3> Ainv_p{};
  solve_linear_system(A, p, Ainv_p);

  Vector<Float, NE> phi{};
  for (Index i = 0; i < NE; ++i) {
    for (Index j = 0; j < 3; ++j) {
      phi(i) += B(j, i) * Ainv_p(j);
    }
  }
  Float res = 0.0;
  for (size_t k = 0; k < neighbours.size(); ++k) {
    const auto [i, j]  = neighbours[k];
    res               += phi(static_cast<Index>(k)) * field(i, j);
  }
  return res;
}

// =================================================================================================
auto main() -> int {
  // = Prepare output directory ====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) {
    Igor::Error("Could not initialize output directory `{}`.", OUTPUT_DIR);
    return 1;
  }

  // = Prepare fluid state =========================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  Igor::Info("x_min = {:.1f}", X_MIN);
  Igor::Info("x_max = {:.1f}", X_MAX);
  Igor::Info("y_min = {:.1f}", Y_MIN);
  Igor::Info("y_max = {:.1f}", Y_MAX);
  Igor::Info("dx    = {:.6e}", fs.dx);
  Igor::Info("dy    = {:.6e}", fs.dy);

  Float t = 0.0;
  for_each_a<Exec::Parallel>(
      fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = u_analytical(fs.x(i), fs.ym(j), t); });
  for_each_a<Exec::Parallel>(
      fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = v_analytical(fs.xm(i), fs.y(j), t); });

  Vector2 point{.x = 3.0, .y = 4.0};
  const auto Up_analytical = u_analytical(point.x, point.y, t);
  const auto Vp_analytical = v_analytical(point.x, point.y, t);
  {
    Igor::Info("Analytical:");
    Igor::Info("  U({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Up_analytical);
    Igor::Info("  V({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Vp_analytical);
    std::cout << '\n';
  }

  {
    const auto Up = bilinear_interpolate(fs.x, fs.ym, fs.curr.U, point.x, point.y);
    const auto Vp = bilinear_interpolate(fs.xm, fs.y, fs.curr.V, point.x, point.y);
    Igor::Info("Bilinear interpolation:");
    Igor::Info("  U({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Up);
    Igor::Info("  V({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Vp);
    Igor::Info(
        "  |U - Ua|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Up - Up_analytical));
    Igor::Info(
        "  |V - Va|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Vp - Vp_analytical));
    std::cout << '\n';
  }

  {
    const auto Up = eval_field_at_nn(fs.x, fs.ym, fs.curr.U, point.x, point.y);
    const auto Vp = eval_field_at_nn(fs.xm, fs.y, fs.curr.V, point.x, point.y);
    Igor::Info("Nearest neighbour:");
    Igor::Info("  U({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Up);
    Igor::Info("  V({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Vp);
    Igor::Info(
        "  |U - Ua|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Up - Up_analytical));
    Igor::Info(
        "  |V - Va|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Vp - Vp_analytical));
    std::cout << '\n';
  }

  {
    const auto Up = eval_field_at_sf(fs.x, fs.ym, fs.curr.U, point.x, point.y);
    const auto Vp = eval_field_at_sf(fs.xm, fs.y, fs.curr.V, point.x, point.y);
    Igor::Info("Shape function:");
    Igor::Info("  U({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Up);
    Igor::Info("  V({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, Vp);
    Igor::Info(
        "  |U - Ua|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Up - Up_analytical));
    Igor::Info(
        "  |V - Va|({:.1f}, {:.1f}) = {:.6e}", point.x, point.y, std::abs(Vp - Vp_analytical));
    std::cout << '\n';
  }
}
