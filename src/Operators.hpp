#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>

#include "Container.hpp"

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
void interpolate_UV_staggered_field(const Matrix<Float, NX + 1, NY>& u_stag,
                                    const Matrix<Float, NX, NY + 1>& v_stag,
                                    Matrix<Float, NX, NY>& interp) noexcept {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      interp[i, j] = (u_stag[i, j] + u_stag[i + 1, j] + v_stag[i, j] + v_stag[i, j + 1]) / 4.0;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_divergence(const Matrix<Float, NX + 1, NY>& U,
                     const Matrix<Float, NX, NY + 1>& V,
                     const Vector<Float, NX>& dx,
                     const Vector<Float, NY>& dy,
                     Matrix<Float, NX, NY>& div) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      div[i, j] = (U[i + 1, j] - U[i, j]) / dx[i] + (V[i, j + 1] - V[i, j]) / dy[j];
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
constexpr auto integrate(const Vector<Float, NX>& dx,
                         const Vector<Float, NY>& dy,
                         const Matrix<Float, NX, NY>& field) noexcept -> Float {
  Float integral = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      integral += field[i, j] * dx[i] * dy[j];
    }
  }
  return integral;
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr auto L1_norm(const Vector<Float, NX>& dx,
                       const Vector<Float, NY>& dy,
                       const Matrix<Float, NX, NY>& field) noexcept -> Float {
  Float integral = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      integral += std::abs(field[i, j]) * dx[i] * dy[j];
    }
  }
  return integral;
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void shift_pressure_to_zero(const Vector<Float, NX>& dx,
                            const Vector<Float, NY>& dy,
                            Matrix<Float, NX, NY>& dp) {
  Float vol_avg_p = integrate(dx, dy, dp);
  for (Index i = 0; i < dp.extent(0); ++i) {
    for (Index j = 0; j < dp.extent(1); ++j) {
      dp[i, j] -= vol_avg_p;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
[[nodiscard]] constexpr auto bilinear_interpolate(const Vector<Float, NX>& xm,
                                                  const Vector<Float, NY>& ym,
                                                  const Matrix<Float, NX, NY>& field,
                                                  Float x,
                                                  Float y) -> Float {
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

  // Interpolate in x
  const auto a =
      (field[inext, jprev] - field[iprev, jprev]) / dx * (x - xm[iprev]) + field[iprev, jprev];
  const auto b =
      (field[inext, jnext] - field[iprev, jnext]) / dx * (x - xm[iprev]) + field[iprev, jnext];

  // Interpolate in y
  return (b - a) / dy * (y - ym[jprev]) + a;
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
    const auto prev = static_cast<Index>(std::floor((pos - grid[0]) / delta));
    const auto next = static_cast<Index>(std::floor((pos - grid[0]) / delta + 1.0));
    if (pos <= grid[0] || prev < 0) { return {0, 0}; }
    if (pos >= grid[N - 1] || next >= N) { return {N - 1, N - 1}; }
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

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_grad_of_centered_points(const Matrix<Float, NX, NY>& f,
                                  Float dx,
                                  Float dy,
                                  Matrix<Float, NX, NY>& dfdx,
                                  Matrix<Float, NX, NY>& dfdy) noexcept {
  // TODO: This assumes an equidistant mesh
  for (Index i = 1; i < NX - 1; ++i) {
    for (Index j = 1; j < NY - 1; ++j) {
      dfdx[i, j] = (f[i + 1, j] - f[i - 1, j]) / (2.0 * dx);
      dfdy[i, j] = (f[i, j + 1] - f[i, j - 1]) / (2.0 * dy);
    }
  }

  for (Index i = 0; i < NX; ++i) {
    if (i > 0 && i < NX - 1) {
      dfdx[i, 0]      = (f[i + 1, 0] - f[i - 1, 0]) / (2.0 * dx);
      dfdx[i, NY - 1] = (f[i + 1, NY - 1] - f[i - 1, NY - 1]) / (2.0 * dx);
    }
    dfdy[i, 0]      = (-3.0 * f[i, 0] + 4.0 * f[i, 1] - f[i, 2]) / (2.0 * dy);
    dfdy[i, NY - 1] = (3.0 * f[i, NY - 1] - 4.0 * f[i, NY - 2] + f[i, NY - 3]) / (2.0 * dy);
  }

  for (Index j = 0; j < NY; ++j) {
    dfdx[0, j]      = (-3.0 * f[0, j] + 4.0 * f[1, j] - f[2, j]) / (2.0 * dx);
    dfdx[NX - 1, j] = (3.0 * f[NX - 1, j] - 4.0 * f[NX - 2, j] + f[NX - 3, j]) / (2.0 * dx);
    if (j > 0 && j < NY - 1) {
      dfdy[0, j]      = (f[0, j + 1] - f[0, j - 1]) / (2.0 * dy);
      dfdy[NX - 1, j] = (f[NX - 1, j + 1] - f[NX - 1, j - 1]) / (2.0 * dy);
    }
  }
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
