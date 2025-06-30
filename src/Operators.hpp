#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>

#include "FS.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void interpolate_U(const Matrix<Float, NX + 1, NY>& U, Matrix<Float, NX, NY>& Ui) {
  for (Index i = 0; i < Ui.extent(0); ++i) {
    for (Index j = 0; j < Ui.extent(1); ++j) {
      Ui[i, j] = (U[i, j] + U[i + 1, j]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void interpolate_V(const Matrix<Float, NX, NY + 1>& V, Matrix<Float, NX, NY>& Vi) {
  for (Index i = 0; i < Vi.extent(0); ++i) {
    for (Index j = 0; j < Vi.extent(1); ++j) {
      Vi[i, j] = (V[i, j] + V[i, j + 1]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_divergence(const FS<Float, NX, NY>& fs, Matrix<Float, NX, NY>& div) {
  for (Index i = 0; i < div.extent(0); ++i) {
    for (Index j = 0; j < div.extent(1); ++j) {
      div[i, j] = (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +  //
                  (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_mid_time(Matrix<Float, NX, NY>& current, const Matrix<Float, NX, NY>& old) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      current[i, j] = 0.5 * (current[i, j] + old[i, j]);
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void shift_pressure_to_zero(const FS<Float, NX, NY>& fs, Matrix<Float, NX, NY>& dp) {
  Float vol_avg_p = 0.0;

  for (Index i = 0; i < dp.extent(0); ++i) {
    for (Index j = 0; j < dp.extent(1); ++j) {
      vol_avg_p += dp[i, j] * fs.dx[i] * fs.dy[j];
    }
  }

  for (Index i = 0; i < dp.extent(0); ++i) {
    for (Index j = 0; j < dp.extent(1); ++j) {
      dp[i, j] -= vol_avg_p;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
[[nodiscard]] constexpr auto eval_flow_field_at(const Vector<Float, NX>& xm,
                                                const Vector<Float, NY>& ym,
                                                const Matrix<Float, NX, NY>& Ui,
                                                const Matrix<Float, NX, NY>& Vi,
                                                Float x,
                                                Float y) -> std::pair<Float, Float> {
  // TODO: Assumes equidistant grid
  const auto dx = xm[1] - xm[0];
  const auto dy = ym[1] - ym[0];

  auto get_indices =
      []<Index N>(Float pos, const Vector<Float, N>& grid, Float delta) -> std::pair<Index, Index> {
    if (pos <= grid[0]) { return {0, 0}; }
    if (pos >= grid[N - 1]) { return {N - 1, N - 1}; }
    const auto prev = static_cast<Index>(std::floor((pos - grid[0]) / delta));
    const auto next = static_cast<Index>(std::floor((pos - grid[0]) / delta + 1.0));
    return {prev, next};
  };

  const auto [iprev, inext] = get_indices(x, xm, dx);
  const auto [jprev, jnext] = get_indices(y, ym, dy);

  auto interpolate_bilinear = [&](const Matrix<Float, NX, NY>& field) -> Float {
    // Interpolate in x
    const auto a =
        (field[inext, jprev] - field[iprev, jprev]) / dx * (x - xm[iprev]) + field[iprev, jprev];
    const auto b =
        (field[inext, jnext] - field[iprev, jnext]) / dx * (x - xm[iprev]) + field[iprev, jnext];

    // Interpolate in y
    return (b - a) / dy * (y - ym[jprev]) + a;
  };

  return {interpolate_bilinear(Ui), interpolate_bilinear(Vi)};
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
