#ifndef FLUID_SOLVER_CONTAINER_HPP_
#define FLUID_SOLVER_CONTAINER_HPP_

#include <cstddef>
#include <memory>

#include <Igor/Logging.hpp>

// =================================================================================================
template <typename Contained, size_t N>
class Vector {
  std::unique_ptr<std::array<Contained, N>> m_data{};

 public:
  Vector()
      : m_data(new std::array<Contained, N>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
  }
  Vector(const Vector& other) noexcept                              = delete;
  Vector(Vector&& other) noexcept                                   = delete;
  constexpr auto operator=(const Vector& other) noexcept -> Vector& = delete;
  constexpr auto operator=(Vector&& other) noexcept -> Vector&      = delete;
  ~Vector() noexcept                                                = default;

  [[nodiscard]] constexpr auto operator[](size_t idx) noexcept -> Contained& {
    IGOR_ASSERT(idx < N, "Index {} is out of bounds for Vector of size {}", idx, N);
    return (*m_data)[idx];
  }

  [[nodiscard]] constexpr auto operator[](size_t idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx < N, "Index {} is out of bounds for Vector of size {}", idx, N);
    return (*m_data)[idx];
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return &((*m_data)[0]); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return &((*m_data)[0]);
  }

  [[nodiscard]] constexpr auto size() const noexcept -> size_t { return N; }

  [[nodiscard]] constexpr auto extent([[maybe_unused]] size_t r) const noexcept -> size_t {
    IGOR_ASSERT(r < 1, "Dimension {} is out of bounds for Vector", r);
    return N;
  }
};

// =================================================================================================
enum class Layout : uint8_t { C, F };
template <typename Contained, size_t M, size_t N, Layout LAYOUT = Layout::C>
class Matrix {
  std::unique_ptr<std::array<Contained, M * N>> m_data{};

  [[nodiscard]] constexpr auto get_idx(size_t i, size_t j) const noexcept -> size_t {
    if constexpr (LAYOUT == Layout::C) {
      return j + i * N;
    } else {
      return i + j * M;
    }
  }

 public:
  Matrix()
      : m_data(new std::array<Contained, M * N>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
  }
  Matrix(const Matrix& other) noexcept                              = delete;
  Matrix(Matrix&& other) noexcept                                   = delete;
  constexpr auto operator=(const Matrix& other) noexcept -> Matrix& = delete;
  constexpr auto operator=(Matrix&& other) noexcept -> Matrix&      = delete;
  ~Matrix() noexcept                                                = default;

  constexpr auto operator[](size_t i, size_t j) noexcept -> Contained& {
    IGOR_ASSERT(
        i < M && j < N, "Index ({}, {}) is out of bounds for Matrix of size {}x{}", i, j, M, N);
    return (*m_data)[get_idx(i, j)];
  }

  constexpr auto operator[](size_t i, size_t j) const noexcept -> const Contained& {
    IGOR_ASSERT(
        i < M && j < N, "Index ({}, {}) is out of bounds for Matrix of size {}x{}", i, j, M, N);
    return (*m_data)[get_idx(i, j)];
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return &((*m_data)[0]); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return &((*m_data)[0]);
  }

  [[nodiscard]] constexpr auto size() const noexcept -> size_t { return M * N; }

  [[nodiscard]] constexpr auto extent(size_t r) const noexcept -> size_t {
    IGOR_ASSERT(r < 2, "Dimension {} is out of bounds for Vector", r);
    return r == 0 ? M : N;
  }
};

#endif  // FLUID_SOLVER_CONTAINER_HPP_
