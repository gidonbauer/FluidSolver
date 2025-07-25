#ifndef FLUID_SOLVER_CONTAINER_HPP_
#define FLUID_SOLVER_CONTAINER_HPP_

#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>

#include <Igor/Logging.hpp>

using Index = int;

// =================================================================================================
template <typename Contained, Index N>
requires(N >= 0)
class Vector {
 public:
  static constexpr bool is_small = N * sizeof(Contained) < 1024UZ;

 private:
  using StorageType = std::
      conditional_t<is_small, std::array<Contained, N>, std::unique_ptr<std::array<Contained, N>>>;
  StorageType m_data{};

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
      : m_data(new std::array<Contained, static_cast<size_t>(N)>) {
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

  [[nodiscard]] constexpr auto operator[](Index idx) noexcept -> Contained& {
    IGOR_ASSERT(idx >= 0 && idx < N, "Index {} is out of bounds for Vector of size {}", idx, N);
    return *(get_data() + idx);
  }

  [[nodiscard]] constexpr auto operator[](Index idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx >= 0 && idx < N, "Index {} is out of bounds for Vector of size {}", idx, N);
    return *(get_data() + idx);
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

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return N; }

  [[nodiscard]] constexpr auto extent([[maybe_unused]] Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 1, "Dimension {} is out of bounds for Vector", r);
    return N;
  }

  [[nodiscard]] constexpr auto begin() noexcept -> Contained* { return get_data(); }
  [[nodiscard]] constexpr auto end() noexcept -> Contained* { return get_data() + size(); }

  [[nodiscard]] constexpr auto is_valid_index(Index i) const noexcept -> bool {
    return 0 <= i && i < N;
  }
};

// =================================================================================================
enum class Layout : uint8_t { C, F };
template <typename Contained, Index M, Index N, Layout LAYOUT = Layout::C>
requires(M >= 0 && N >= 0)
class Matrix {
 public:
  static constexpr bool is_small = M * N * sizeof(Contained) < 1024UZ;

 private:
  using StorageType = std::conditional_t<is_small,
                                         std::array<Contained, M * N>,
                                         std::unique_ptr<std::array<Contained, M * N>>>;
  StorageType m_data{};

  [[nodiscard]] constexpr auto get_idx(Index i, Index j) const noexcept -> Index {
    if constexpr (LAYOUT == Layout::C) {
      return j + i * N;
    } else {
      return i + j * M;
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
      : m_data(new std::array<Contained, M * N>) {
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

  constexpr auto operator[](Index i, Index j) noexcept -> Contained& {
    IGOR_ASSERT(i >= 0 && i < M && j >= 0 && j < N,
                "Index ({}, {}) is out of bounds for Matrix of size {}x{}",
                i,
                j,
                M,
                N);
    return *(get_data() + get_idx(i, j));
  }

  constexpr auto operator[](Index i, Index j) const noexcept -> const Contained& {
    IGOR_ASSERT(i >= 0 && i < M && j >= 0 && j < N,
                "Index ({}, {}) is out of bounds for Matrix of size {}x{}",
                i,
                j,
                M,
                N);
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

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return M * N; }

  [[nodiscard]] constexpr auto extent(Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 2, "Dimension {} is out of bounds for Vector", r);
    return r == 0 ? M : N;
  }

  [[nodiscard]] constexpr auto is_valid_index(Index i, Index j) const noexcept -> bool {
    return 0 <= i && i < M && 0 <= j && j < N;
  }
};

// -------------------------------------------------------------------------------------------------
template <typename CT>
[[nodiscard]] constexpr auto max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return std::transform_reduce(
      c.get_data(),
      c.get_data() + c.size(),
      -std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });
}

// -------------------------------------------------------------------------------------------------
template <typename CT>
[[nodiscard]] constexpr auto min(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return std::transform_reduce(
      c.get_data(),
      c.get_data() + c.size(),
      std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::min(a, b); },
      [](Float x) { return std::abs(x); });
}

#endif  // FLUID_SOLVER_CONTAINER_HPP_
