#ifndef FLUID_SOLVER_CONTAINER_HPP_
#define FLUID_SOLVER_CONTAINER_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>

#include <Igor/Logging.hpp>

#ifndef FS_INDEX_TYPE
using Index = int;
#else
static_assert(std::is_integral_v<FS_INDEX_TYPE> && std::is_signed_v<FS_INDEX_TYPE>,
              "FS_INDEX_TYPE must be a signed integer type.");
using Index = FS_INDEX_TYPE;
#endif  // FS_INDEX_TYPE

enum class Layout : uint8_t { C, F };

// =================================================================================================
template <typename Contained, Index N, Index NGHOST = 0>
requires(N > 0 && NGHOST >= 0)
class Vector {
  static constexpr auto ARRAY_SIZE = static_cast<size_t>(N + 2 * NGHOST);

 public:
  static constexpr bool is_small = N <= 5 && ARRAY_SIZE * sizeof(Contained) < 1024UZ;

 private:
  using StorageType = std::conditional_t<is_small,
                                         std::array<Contained, ARRAY_SIZE>,
                                         std::unique_ptr<std::array<Contained, ARRAY_SIZE>>>;
  StorageType m_data{};

  [[nodiscard]] constexpr auto get_idx(Index raw_idx) const noexcept -> Index {
    return raw_idx + NGHOST;
  }

 public:
  Vector() noexcept
  requires(is_small)
  {
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data.data(), size(), Contained{0});
    }
  }
  Vector() noexcept
  requires(!is_small)
      : m_data(new std::array<Contained, ARRAY_SIZE>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data->data(), size(), Contained{0});
    }
  }
  Vector(const Vector& other) noexcept                              = delete;
  Vector(Vector&& other) noexcept                                   = delete;
  constexpr auto operator=(const Vector& other) noexcept -> Vector& = delete;
  constexpr auto operator=(Vector&& other) noexcept -> Vector&      = delete;
  ~Vector() noexcept                                                = default;

  [[nodiscard]] constexpr auto operator()(Index idx) noexcept -> Contained& {
    IGOR_ASSERT(idx >= -NGHOST && idx < N + NGHOST,
                "Index {} is out of bounds for Vector with dimension {}:{}",
                idx,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(idx));
  }

  [[nodiscard]] constexpr auto operator()(Index idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx >= -NGHOST && idx < N + NGHOST,
                "Index {} is out of bounds for Vector with dimension {}:{}",
                idx,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(idx));
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* {
    if constexpr (is_small) {
      return m_data.data();
    } else {
      return m_data->data();
    }
  }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    if constexpr (is_small) {
      return m_data.data();
    } else {
      return m_data->data();
    }
  }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return ARRAY_SIZE; }

  [[nodiscard]] constexpr auto n_ghost() const noexcept -> Index { return NGHOST; }

  [[nodiscard]] constexpr auto extent([[maybe_unused]] Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 1, "Dimension {} is out of bounds for Vector", r);
    return N;
  }

  // TODO: Shoud this exist and should it include ghost cells?
  // [[nodiscard]] constexpr auto begin() noexcept -> Contained* { return get_data(); }
  // [[nodiscard]] constexpr auto end() noexcept -> Contained* { return get_data() + size(); }

  [[nodiscard]] constexpr auto is_valid_interior_index(Index i) const noexcept -> bool {
    return 0 <= i && i < N;
  }
  [[nodiscard]] constexpr auto is_valid_index(Index i) const noexcept -> bool {
    return -NGHOST <= i && i < N + NGHOST;
  }
};

// =================================================================================================
template <typename Contained, Index M, Index N, Index NGHOST = 0, Layout LAYOUT = Layout::C>
requires(M > 0 && N > 0 && NGHOST >= 0)
class Matrix {
  static constexpr auto ARRAY_SIZE = (M + 2 * NGHOST) * (N + 2 * NGHOST);

 public:
  static constexpr bool is_small = M <= 5 && N <= 5 && ARRAY_SIZE * sizeof(Contained) < 1024UZ;

 private:
  using StorageType = std::conditional_t<is_small,
                                         std::array<Contained, ARRAY_SIZE>,
                                         std::unique_ptr<std::array<Contained, ARRAY_SIZE>>>;
  StorageType m_data{};

  [[nodiscard]] constexpr auto get_idx(Index i, Index j) const noexcept -> Index {
    if constexpr (LAYOUT == Layout::C) {
      return (j + NGHOST) + (i + NGHOST) * (N + 2 * NGHOST);
    } else {
      return (i + NGHOST) + (j + NGHOST) * (M + 2 * NGHOST);
    }
  }

 public:
  Matrix() noexcept
  requires(is_small)
  {
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data.data(), size(), Contained{0});
    }
  }
  Matrix() noexcept
  requires(!is_small)
      : m_data(new std::array<Contained, ARRAY_SIZE>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data->data(), size(), Contained{0});
    }
  }
  Matrix(const Matrix& other) noexcept                              = delete;
  Matrix(Matrix&& other) noexcept                                   = delete;
  constexpr auto operator=(const Matrix& other) noexcept -> Matrix& = delete;
  constexpr auto operator=(Matrix&& other) noexcept -> Matrix&      = delete;
  ~Matrix() noexcept                                                = default;

  constexpr auto operator()(Index i, Index j) noexcept -> Contained& {
    IGOR_ASSERT(i >= -NGHOST && i < M + NGHOST && j >= -NGHOST && j < N + NGHOST,
                "Index ({}, {}) is out of bounds for Matrix of size {}:{}x{}:{}",
                i,
                j,
                -NGHOST,
                M + NGHOST,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(i, j));
  }

  constexpr auto operator()(Index i, Index j) const noexcept -> const Contained& {
    IGOR_ASSERT(i >= -NGHOST && i < M + NGHOST && j >= -NGHOST && j < N + NGHOST,
                "Index ({}, {}) is out of bounds for Matrix of size {}:{}x{}:{}",
                i,
                j,
                -NGHOST,
                M + NGHOST,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(i, j));
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* {
    if constexpr (is_small) {
      return m_data.data();
    } else {
      return m_data->data();
    }
  }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    if constexpr (is_small) {
      return m_data.data();
    } else {
      return m_data->data();
    }
  }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return ARRAY_SIZE; }
  [[nodiscard]] constexpr auto n_ghost() const noexcept -> Index { return NGHOST; }

  [[nodiscard]] constexpr auto extent(Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 2, "Dimension {} is out of bounds for Vector", r);
    return r == 0 ? M : N;
  }

  [[nodiscard]] constexpr auto is_valid_interior_index(Index i, Index j) const noexcept -> bool {
    return 0 <= i && i < M && 0 <= j && j < N;
  }
  [[nodiscard]] constexpr auto is_valid_index(Index i, Index j) const noexcept -> bool {
    return -NGHOST <= i && i < M + NGHOST && -NGHOST <= j && j < N + NGHOST;
  }
};

// -------------------------------------------------------------------------------------------------
template <typename CT>
[[nodiscard]] constexpr auto abs_max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return std::transform_reduce(
      c.get_data(),
      c.get_data() + c.size(),
      std::abs(*c.get_data()),
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });
}

// -------------------------------------------------------------------------------------------------
template <typename CT>
[[nodiscard]] constexpr auto max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return std::reduce(c.get_data(), c.get_data() + c.size(), *c.get_data(), [](Float a, Float b) {
    return std::max(a, b);
  });
}

// -------------------------------------------------------------------------------------------------
template <typename CT>
[[nodiscard]] constexpr auto min(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return std::reduce(c.get_data(), c.get_data() + c.size(), *c.get_data(), [](Float a, Float b) {
    return std::min(a, b);
  });
}

// -------------------------------------------------------------------------------------------------
template <typename CT, typename Float>
constexpr void fill(CT& c, Float value) noexcept {
  static_assert(std::is_same_v<Float, std::remove_cvref_t<decltype(*c.get_data())>>,
                "Incompatible type of value and Contained");
  std::fill_n(c.get_data(), c.size(), value);
}

// -------------------------------------------------------------------------------------------------
template <typename CT>
constexpr void copy(const CT& src, CT& dst) noexcept {
  std::copy_n(src.get_data(), src.size(), dst.get_data());
}

// -------------------------------------------------------------------------------------------------
template <typename CT, typename Pred>
constexpr auto any(const CT& c, Pred&& pred) noexcept -> bool {
  return std::any_of(c.get_data(), c.get_data() + c.size(), std::forward<Pred&&>(pred));
}

template <typename CT>
constexpr auto has_nan(const CT& c) noexcept -> bool {
  return any(c, [](auto val) { return std::isnan(val); });
}

template <typename CT>
constexpr auto has_inf(const CT& c) noexcept -> bool {
  return any(c, [](auto val) { return std::isinf(val); });
}

template <typename CT>
constexpr auto has_nan_or_inf(const CT& c) noexcept -> bool {
  return has_nan(c) || has_inf(c);
}

#endif  // FLUID_SOLVER_CONTAINER_HPP_
