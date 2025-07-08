#ifndef FLUID_SOLVER_CONTAINER_HPP_
#define FLUID_SOLVER_CONTAINER_HPP_

#include <cstddef>
#include <memory>

#include <Igor/Logging.hpp>

using Index = int;

// =================================================================================================
template <typename Contained, Index N>
requires(N >= 0)
class Vector {
  std::unique_ptr<std::array<Contained, N>> m_data{};

 public:
  Vector() noexcept
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
    return *(m_data->data() + idx);
  }

  [[nodiscard]] constexpr auto operator[](Index idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx >= 0 && idx < N, "Index {} is out of bounds for Vector of size {}", idx, N);
    return *(m_data->data() + idx);
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return m_data->data(); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return m_data->data();
  }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return N; }

  [[nodiscard]] constexpr auto extent([[maybe_unused]] Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 1, "Dimension {} is out of bounds for Vector", r);
    return N;
  }

  [[nodiscard]] constexpr auto begin() noexcept -> Contained* { return m_data->data(); }
  [[nodiscard]] constexpr auto end() noexcept -> Contained* { return m_data->data() + size(); }

  [[nodiscard]] constexpr auto is_valid_index(Index i) const noexcept -> bool {
    return 0 <= i && i < N;
  }
};

// =================================================================================================
enum class Layout : uint8_t { C, F };
template <typename Contained, Index M, Index N, Layout LAYOUT = Layout::C>
requires(M >= 0 && N >= 0)
class Matrix {
  std::unique_ptr<std::array<Contained, static_cast<size_t>(M* N)>> m_data{};

  [[nodiscard]] constexpr auto get_idx(Index i, Index j) const noexcept -> Index {
    if constexpr (LAYOUT == Layout::C) {
      return j + i * N;
    } else {
      return i + j * M;
    }
  }

 public:
  Matrix() noexcept
      : m_data(new std::array<Contained, static_cast<size_t>(M* N)>) {
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
    return *(m_data->data() + get_idx(i, j));
  }

  constexpr auto operator[](Index i, Index j) const noexcept -> const Contained& {
    IGOR_ASSERT(i >= 0 && i < M && j >= 0 && j < N,
                "Index ({}, {}) is out of bounds for Matrix of size {}x{}",
                i,
                j,
                M,
                N);
    return *(m_data->data() + get_idx(i, j));
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return m_data->data(); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return m_data->data();
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

#endif  // FLUID_SOLVER_CONTAINER_HPP_
