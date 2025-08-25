#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "ForEach.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_U(const Matrix<Float, NX + 1, NY, NGHOST>& U, Matrix<Float, NX, NY, NGHOST>& Ui) {
  for_each_a<Exec::Parallel>(Ui, [&](Index i, Index j) { Ui(i, j) = (U(i, j) + U(i + 1, j)) / 2; });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_V(const Matrix<Float, NX, NY + 1, NGHOST>& V, Matrix<Float, NX, NY, NGHOST>& Vi) {
  for_each_a<Exec::Parallel>(Vi, [&](Index i, Index j) { Vi(i, j) = (V(i, j) + V(i, j + 1)) / 2; });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void interpolate_UV_staggered_field(const Matrix<Float, NX + 1, NY, NGHOST>& u_stag,
                                    const Matrix<Float, NX, NY + 1, NGHOST>& v_stag,
                                    Matrix<Float, NX, NY, NGHOST>& interp) noexcept {
  for_each_a<Exec::Parallel>(interp, [&](Index i, Index j) {
    interp(i, j) = (u_stag(i, j) + u_stag(i + 1, j) + v_stag(i, j) + v_stag(i, j + 1)) / 4.0;
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_divergence(const Matrix<Float, NX + 1, NY, NGHOST>& U,
                     const Matrix<Float, NX, NY + 1, NGHOST>& V,
                     Float dx,
                     Float dy,
                     Matrix<Float, NX, NY, NGHOST>& div) {
  for_each_a<Exec::Parallel>(div, [&](Index i, Index j) {
    div(i, j) = (U(i + 1, j) - U(i, j)) / dx + (V(i, j + 1) - V(i, j)) / dy;
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_mid_time(Matrix<Float, NX, NY, NGHOST>& current,
                   const Matrix<Float, NX, NY, NGHOST>& old) {
  for_each_a<Exec::Parallel>(
      current, [&](Index i, Index j) { current(i, j) = 0.5 * (current(i, j) + old(i, j)); });
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto integrate(Float dx, Float dy, const Matrix<Float, NX, NY, NGHOST>& field) noexcept
    -> Float {
  Float integral = 0.0;
  if (INCLUDE_GHOST) {
    for_each_a(field, [&](Index i, Index j) { integral += field(i, j); });
  } else {
    for_each_i(field, [&](Index i, Index j) { integral += field(i, j); });
  }
  return integral * dx * dy;
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto L1_norm(Float dx, Float dy, const Matrix<Float, NX, NY, NGHOST>& field) noexcept
    -> Float {
  Float integral = 0.0;
  if (INCLUDE_GHOST) {
    for_each_a(field, [&](Index i, Index j) { integral += std::abs(field(i, j)); });
  } else {
    for_each_i(field, [&](Index i, Index j) { integral += std::abs(field(i, j)); });
  }
  return integral * dx * dy;
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void shift_pressure_to_zero(Float dx, Float dy, Matrix<Float, NX, NY, NGHOST>& dp) {
  Float vol_avg_p = integrate<true>(dx, dy, dp);
  for_each_a<Exec::Parallel>(dp, [&](Index i, Index j) { dp(i, j) -= vol_avg_p; });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
[[nodiscard]] constexpr auto bilinear_interpolate(const Vector<Float, NX, NGHOST>& xm,
                                                  const Vector<Float, NY, NGHOST>& ym,
                                                  const Matrix<Float, NX, NY, NGHOST>& field,
                                                  Float x,
                                                  Float y) -> Float {
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

  const auto [iprev, inext] = get_indices(x, xm, dx);
  const auto [jprev, jnext] = get_indices(y, ym, dy);

  // Interpolate in x
  const auto a =
      (field(inext, jprev) - field(iprev, jprev)) / dx * (x - xm(iprev)) + field(iprev, jprev);
  const auto b =
      (field(inext, jnext) - field(iprev, jnext)) / dx * (x - xm(iprev)) + field(iprev, jnext);

  // Interpolate in y
  return (b - a) / dy * (y - ym(jprev)) + a;
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
[[nodiscard]] constexpr auto eval_flow_field_at(const Vector<Float, NX, NGHOST>& xm,
                                                const Vector<Float, NY, NGHOST>& ym,
                                                const Matrix<Float, NX, NY, NGHOST>& Ui,
                                                const Matrix<Float, NX, NY, NGHOST>& Vi,
                                                Float x,
                                                Float y) -> std::pair<Float, Float> {
  const auto dx    = xm(1) - xm(0);
  const auto dy    = ym(1) - ym(0);

  auto get_indices = []<Index N>(Float pos,
                                 const Vector<Float, N, NGHOST>& grid,
                                 Float delta) -> std::pair<Index, Index> {
    const auto prev = static_cast<Index>(std::floor((pos - grid(0)) / delta));
    const auto next = static_cast<Index>(std::floor((pos - grid(0)) / delta + 1.0));
    if (pos <= grid(0) || prev < 0) { return {0, 0}; }
    if (pos >= grid(N - 1) || next >= N) { return {N - 1, N - 1}; }
    return {prev, next};
  };

  // const auto [iprev, inext] = get_indices(x, xm, dx);
  // const auto [jprev, jnext] = get_indices(y, ym, dy);

  const auto i_pair         = get_indices(x, xm, dx);
  const auto j_pair         = get_indices(y, ym, dy);

  const auto iprev          = i_pair.first;
  const auto inext          = i_pair.second;
  const auto jprev          = j_pair.first;
  const auto jnext          = j_pair.second;

  auto interpolate_bilinear = [&](const Matrix<Float, NX, NY, NGHOST>& field) -> Float {
    // Interpolate in x
    const auto a =
        (field(inext, jprev) - field(iprev, jprev)) / dx * (x - xm(iprev)) + field(iprev, jprev);
    const auto b =
        (field(inext, jnext) - field(iprev, jnext)) / dx * (x - xm(iprev)) + field(iprev, jnext);

    // Interpolate in y
    return (b - a) / dy * (y - ym(jprev)) + a;
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
  for_each<-NGHOST + 1, NX + NGHOST - 1, -NGHOST + 1, NY + NGHOST - 1, Exec::Parallel>(
      [&](Index i, Index j) {
        dfdx(i, j) = (f(i + 1, j) - f(i - 1, j)) / (2.0 * dx);
        dfdy(i, j) = (f(i, j + 1) - f(i, j - 1)) / (2.0 * dy);
      });

  for_each<-NGHOST, NX + NGHOST, Exec::Parallel>([&](Index i) {
    if (i > -NGHOST && i < NX + NGHOST - 1) {
      dfdx(i, -NGHOST) = (f(i + 1, -NGHOST) - f(i - 1, -NGHOST)) / (2.0 * dx);
      dfdx(i, NY + NGHOST - 1) =
          (f(i + 1, NY + NGHOST - 1) - f(i - 1, NY + NGHOST - 1)) / (2.0 * dx);
    }
    dfdy(i, -NGHOST) =
        (-3.0 * f(i, -NGHOST) + 4.0 * f(i, -NGHOST + 1) - f(i, -NGHOST + 2)) / (2.0 * dy);
    dfdy(i, NY + NGHOST - 1) =
        (3.0 * f(i, NY + NGHOST - 1) - 4.0 * f(i, NY + NGHOST - 2) + f(i, NY + NGHOST - 3)) /
        (2.0 * dy);
  });

  for_each<-NGHOST, NY + NGHOST, Exec::Parallel>([&](Index j) {
    dfdx(-NGHOST, j) =
        (-3.0 * f(-NGHOST, j) + 4.0 * f(-NGHOST + 1, j) - f(-NGHOST + 2, j)) / (2.0 * dx);
    dfdx(NX + NGHOST - 1, j) =
        (3.0 * f(NX + NGHOST - 1, j) - 4.0 * f(NX + NGHOST - 2, j) + f(NX + NGHOST - 3, j)) /
        (2.0 * dx);
    if (j > -NGHOST && j < NY + NGHOST - 1) {
      dfdy(-NGHOST, j) = (f(-NGHOST, j + 1) - f(-NGHOST, j - 1)) / (2.0 * dy);
      dfdy(NX + NGHOST - 1, j) =
          (f(NX + NGHOST - 1, j + 1) - f(NX + NGHOST - 1, j - 1)) / (2.0 * dy);
    }
  });
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
