#ifndef FLUID_SOLVER_UTILITY_HPP_
#define FLUID_SOLVER_UTILITY_HPP_

#include <atomic>

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "ForEach.hpp"

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false,
          typename Float,
          Index N,
          Index NGHOST,
          typename BinaryFunc,
          typename UnaryFunc>
[[nodiscard]] constexpr auto transform_reduce(const Vector<Float, N, NGHOST>& vec,
                                              Float init,
                                              BinaryFunc&& reduce,
                                              UnaryFunc&& transform) noexcept -> Float {

  Float res = init;
  if (INCLUDE_GHOST) {
    for_each_a(vec,
               [&res,
                &vec,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i) {
                 res = reduce(res, transform(vec(i)));
               });
  } else {
    for_each_i(vec,
               [&res,
                &vec,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i) {
                 res = reduce(res, transform(vec(i)));
               });
  }
  return res;
}

template <bool INCLUDE_GHOST = false,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          Layout LAYOUT,
          typename BinaryFunc,
          typename UnaryFunc>
[[nodiscard]] constexpr auto transform_reduce(const Matrix<Float, NX, NY, NGHOST, LAYOUT>& mat,
                                              Float init,
                                              BinaryFunc&& reduce,
                                              UnaryFunc&& transform) noexcept -> Float {

  Float res = init;
  if (INCLUDE_GHOST) {
    for_each_a(mat,
               [&mat,
                &res,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i, Index j) {
                 res = reduce(res, transform(mat(i, j)));
               });
  } else {
    for_each_i(mat,
               [&mat,
                &res,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i, Index j) {
                 res = reduce(res, transform(mat(i, j)));
               });
  }
  return res;
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto abs_max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  return transform_reduce<INCLUDE_GHOST>(
      c,
      Float{0},
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return transform_reduce<INCLUDE_GHOST>(
      c,
      -std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return x; });
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto min(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return transform_reduce<INCLUDE_GHOST>(
      c,
      std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::min(a, b); },
      [](Float x) { return x; });
}

// -------------------------------------------------------------------------------------------------
template <typename T>
void update_maximum_atomic(std::atomic<T>& maximum_value, T const& value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value && !maximum_value.compare_exchange_weak(prev_value, value)) {}
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index N>
void solve_linear_system(const Matrix<Float, N, N>& lhs_,
                         const Vector<Float, N>& rhs_,
                         Vector<Float, N>& sol) noexcept {
  Matrix<Float, N, N> lhs{};
  std::copy_n(lhs_.get_data(), lhs_.size(), lhs.get_data());
  Vector<Float, N> rhs{};
  std::copy_n(rhs_.get_data(), rhs_.size(), rhs.get_data());

  auto swap_rows = [&lhs, &rhs](Index a, Index b) {
    if (a == b) { return; }
    for (Index j = 0; j < N; ++j) {
      std::swap(lhs(a, j), lhs(b, j));
    }
    std::swap(rhs(a), rhs(b));
  };

  // = Gaussian Elimination ========================================================================
  for (Index k = 0; k < N - 1; ++k) {
    // - Find best row ---------------------------
    Float max     = std::abs(lhs(k, k));
    Index max_idx = k;
    for (Index i = k + 1; i < N; ++i) {
      if (std::abs(lhs(i, k)) > max) {
        max     = std::abs(lhs(i, k));
        max_idx = i;
      }
    }
    swap_rows(k, max_idx);
    // - Find best row ---------------------------

    for (Index i = k + 1; i < N; ++i) {
      const Float factor = lhs(i, k) / lhs(k, k);
      lhs(i, k)          = 0.0;
      for (Index j = k + 1; j < N; ++j) {
        lhs(i, j) -= factor * lhs(k, j);
      }
      rhs(i) -= factor * rhs(k);
    }
  }

  for (Index i = N - 1; i >= 0; --i) {
    sol(i) = rhs(i);
    for (Index j = N - 1; j > i; --j) {
      sol(i) -= lhs(i, j) * sol(j);
    }
    sol(i) /= lhs(i, i);
  }
  // = Gaussian Elimination ========================================================================
}

// -------------------------------------------------------------------------------------------------
template <typename Float>
void solve_linear_system_explicit(const Matrix<Float, 3, 3>& lhs,
                                  const Vector<Float, 3>& rhs,
                                  Vector<Float, 3>& sol) noexcept {
  Matrix<Float, 3, 3> inv_lhs{};
  inv_lhs(0, 0) = (lhs(1, 1) * lhs(2, 2) - lhs(1, 2) * lhs(2, 1));
  inv_lhs(1, 0) = -(lhs(1, 0) * lhs(2, 2) - lhs(1, 2) * lhs(2, 0));
  inv_lhs(2, 0) = (lhs(1, 0) * lhs(2, 1) - lhs(1, 1) * lhs(2, 0));

  inv_lhs(0, 1) = -(lhs(0, 1) * lhs(2, 2) - lhs(0, 2) * lhs(2, 1));
  inv_lhs(1, 1) = (lhs(0, 0) * lhs(2, 2) - lhs(0, 2) * lhs(2, 0));
  inv_lhs(2, 1) = -(lhs(0, 0) * lhs(2, 1) - lhs(0, 1) * lhs(2, 0));

  inv_lhs(0, 2) = (lhs(0, 1) * lhs(1, 2) - lhs(0, 2) * lhs(1, 1));
  inv_lhs(1, 2) = -(lhs(0, 0) * lhs(1, 2) - lhs(0, 2) * lhs(1, 0));
  inv_lhs(2, 2) = (lhs(0, 0) * lhs(1, 1) - lhs(0, 1) * lhs(1, 0));

  const auto det_lhs =
      lhs(0, 0) * inv_lhs(0, 0) + lhs(0, 1) * inv_lhs(1, 0) + lhs(0, 2) * inv_lhs(2, 0);
  IGOR_ASSERT(
      std::abs(det_lhs) > 1e-8, "Expected invertible matrix but determinant is {:.6e}", det_lhs);
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      inv_lhs(i, j) /= det_lhs;
    }
  }

  for (Index i = 0; i < 3; ++i) {
    sol(i) = 0.0;
    for (Index j = 0; j < 3; ++j) {
      sol(i) += inv_lhs(i, j) * rhs(j);
    }
  }
}

#endif  // FLUID_SOLVER_UTILITY_HPP_
