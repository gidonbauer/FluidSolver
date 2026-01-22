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
[[nodiscard]] constexpr auto transform_reduce(const Field1D<Float, N, NGHOST>& vec,
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
[[nodiscard]] constexpr auto transform_reduce(const Field2D<Float, NX, NY, NGHOST, LAYOUT>& mat,
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
template <std::floating_point Float, Index M, Index N>
requires(M > 0 && N > 0)
class Matrix {
  std::array<Float, M * N> m_data{};

  [[nodiscard]] constexpr auto get_idx(Index i, Index j) const noexcept -> size_t {
    return static_cast<size_t>(j + i * N);
  }

 public:
  constexpr auto operator()(Index i, Index j) noexcept -> Float& {
    IGOR_ASSERT(i >= 0 && i < M && j >= 0 && j < N,
                "Index ({}, {}) is out of bounds for Matrix of size {}x{}",
                i,
                j,
                M,
                N);
    return m_data[get_idx(i, j)];
  }

  constexpr auto operator()(Index i, Index j) const noexcept -> const Float& {
    IGOR_ASSERT(i >= 0 && i < M && j >= 0 && j < N,
                "Index ({}, {}) is out of bounds for Matrix of size {}x{}",
                i,
                j,
                M,
                N);
    return m_data[get_idx(i, j)];
  }
};

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index N>
requires(N > 0)
class Vector {
 public:
  std::array<Float, N> m_data{};

  constexpr auto operator()(Index i) noexcept -> Float& {
    IGOR_ASSERT(i >= 0 && i < N, "Index {} is out of bounds for Vector of size {}", i, N);
    return m_data[static_cast<size_t>(i)];
  }

  constexpr auto operator()(Index i) const noexcept -> const Float& {
    IGOR_ASSERT(i >= 0 && i < N, "Index {} is out of bounds for Vector of size {}", i, N);
    return m_data[static_cast<size_t>(i)];
  }
};

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index N>
requires(N > 0)
constexpr void dot(const Vector<Float, N>& lhs, const Vector<Float, N>& rhs, Float& sol) noexcept {
  sol = 0.0;
  for (Index i = 0; i < N; ++i) {
    sol += lhs(i) * rhs(i);
  }
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index M, Index N>
requires(M > 0 && N > 0)
constexpr void matvecmul(const Matrix<Float, M, N>& lhs,
                         const Vector<Float, N>& rhs,
                         Vector<Float, M>& sol) noexcept {
  for (Index i = 0; i < M; ++i) {
    sol(i) = 0.0;
    for (Index j = 0; j < N; ++j) {
      sol(i) += lhs(i, j) * rhs(j);
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index M, Index K, Index N>
requires(M > 0 && K > 0 && N > 0)
constexpr void matmul(const Matrix<Float, M, K>& lhs,
                      const Matrix<Float, K, N>& rhs,
                      Matrix<Float, M, N>& sol) noexcept {
  for (Index i = 0; i < M; ++i) {
    for (Index j = 0; j < N; ++j) {
      sol(i, j) = 0.0;
      for (Index k = 0; k < K; ++k) {
        sol(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index N>
void solve_linear_system(Matrix<Float, N, N> lhs,
                         Vector<Float, N> rhs,
                         Vector<Float, N>& sol) noexcept {
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
