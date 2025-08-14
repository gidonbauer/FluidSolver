#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>

#include "Container.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_U(const Matrix<Float, NX + 1, NY, NGHOST>& U, Matrix<Float, NX, NY, NGHOST>& Ui) {
  for (Index i = 0; i < Ui.extent(0); ++i) {
    for (Index j = 0; j < Ui.extent(1); ++j) {
      Ui[i, j] = (U[i, j] + U[i + 1, j]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_V(const Matrix<Float, NX, NY + 1, NGHOST>& V, Matrix<Float, NX, NY, NGHOST>& Vi) {
  for (Index i = 0; i < Vi.extent(0); ++i) {
    for (Index j = 0; j < Vi.extent(1); ++j) {
      Vi[i, j] = (V[i, j] + V[i, j + 1]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_UV_staggered_field(const Matrix<Float, NX + 1, NY, NGHOST>& u_stag,
                                    const Matrix<Float, NX, NY + 1, NGHOST>& v_stag,
                                    Matrix<Float, NX, NY, NGHOST>& interp) noexcept {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      interp[i, j] = (u_stag[i, j] + u_stag[i + 1, j] + v_stag[i, j] + v_stag[i, j + 1]) / 4.0;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_divergence(const Matrix<Float, NX + 1, NY, NGHOST>& U,
                     const Matrix<Float, NX, NY + 1, NGHOST>& V,
                     Float dx,
                     Float dy,
                     Matrix<Float, NX, NY, NGHOST>& div) {
  for (Index i = -NGHOST; i < NX + NGHOST; ++i) {
    for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
      div[i, j] = (U[i + 1, j] - U[i, j]) / dx + (V[i, j + 1] - V[i, j]) / dy;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_mid_time(Matrix<Float, NX, NY, NGHOST>& current,
                   const Matrix<Float, NX, NY, NGHOST>& old) {
  for (Index i = -NGHOST; i < NX + NGHOST; ++i) {
    for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
      current[i, j] = 0.5 * (current[i, j] + old[i, j]);
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto integrate(Float dx, Float dy, const Matrix<Float, NX, NY, NGHOST>& field) noexcept
    -> Float {
  constexpr Index I_MIN = INCLUDE_GHOST ? -NGHOST : 0;
  constexpr Index J_MIN = INCLUDE_GHOST ? -NGHOST : 0;
  constexpr Index I_MAX = INCLUDE_GHOST ? NX + NGHOST : NX;
  constexpr Index J_MAX = INCLUDE_GHOST ? NY + NGHOST : NY;

  Float integral        = 0.0;
  for (Index i = I_MIN; i < I_MAX; ++i) {
    for (Index j = J_MIN; j < J_MAX; ++j) {
      integral += field[i, j];
    }
  }
  return integral * dx * dy;
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto L1_norm(Float dx, Float dy, const Matrix<Float, NX, NY, NGHOST>& field) noexcept
    -> Float {
  constexpr Index I_MIN = INCLUDE_GHOST ? -NGHOST : 0;
  constexpr Index J_MIN = INCLUDE_GHOST ? -NGHOST : 0;
  constexpr Index I_MAX = INCLUDE_GHOST ? NX + NGHOST : NX;
  constexpr Index J_MAX = INCLUDE_GHOST ? NY + NGHOST : NY;

  Float integral        = 0.0;
  for (Index i = I_MIN; i < I_MAX; ++i) {
    for (Index j = J_MIN; j < J_MAX; ++j) {
      integral += std::abs(field[i, j]);
    }
  }
  return integral * dx * dy;
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void shift_pressure_to_zero(Float dx, Float dy, Matrix<Float, NX, NY, NGHOST>& dp) {
  Float vol_avg_p = integrate<true>(dx, dy, dp);
  for (Index i = -NGHOST; i < dp.extent(0) + NGHOST; ++i) {
    for (Index j = -NGHOST; j < dp.extent(1) + NGHOST; ++j) {
      dp[i, j] -= vol_avg_p;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
[[nodiscard]] constexpr auto bilinear_interpolate(const Vector<Float, NX, NGHOST>& xm,
                                                  const Vector<Float, NY, NGHOST>& ym,
                                                  const Matrix<Float, NX, NY, NGHOST>& field,
                                                  Float x,
                                                  Float y) -> Float {
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
template <typename Float, Index NX, Index NY, Index NGHOST>
[[nodiscard]] constexpr auto eval_flow_field_at(const Vector<Float, NX, NGHOST>& xm,
                                                const Vector<Float, NY, NGHOST>& ym,
                                                const Matrix<Float, NX, NY, NGHOST>& Ui,
                                                const Matrix<Float, NX, NY, NGHOST>& Vi,
                                                Float x,
                                                Float y) -> std::pair<Float, Float> {
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
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_grad_of_centered_points(const Matrix<Float, NX, NY, NGHOST>& f,
                                  Float dx,
                                  Float dy,
                                  Matrix<Float, NX, NY, NGHOST>& dfdx,
                                  Matrix<Float, NX, NY, NGHOST>& dfdy) noexcept {
  if constexpr (NGHOST == 0) {
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
  } else {
    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        dfdx[i, j] = (f[i + 1, j] - f[i - 1, j]) / (2.0 * dx);
        dfdy[i, j] = (f[i, j + 1] - f[i, j - 1]) / (2.0 * dy);
      }
    }
  }
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
