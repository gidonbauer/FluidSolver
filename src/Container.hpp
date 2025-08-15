#ifndef FLUID_SOLVER_CONTAINER_HPP_
#define FLUID_SOLVER_CONTAINER_HPP_

#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>

#include <Igor/Logging.hpp>

using Index = int;

// =================================================================================================
template <typename Contained, Index N, Index NGHOST = 0>
requires(N >= 0 && NGHOST >= 0)
class Vector {
  static constexpr auto ARRAY_SIZE = static_cast<size_t>(N + 2 * NGHOST);

 public:
  static constexpr bool is_small = ARRAY_SIZE * sizeof(Contained) < 1024UZ;

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

  [[nodiscard]] constexpr auto operator[](Index idx) noexcept -> Contained& {
    IGOR_ASSERT(idx >= -NGHOST && idx < N + NGHOST,
                "Index {} is out of bounds for Vector with dimension {}:{}",
                idx,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(idx));
  }

  [[nodiscard]] constexpr auto operator[](Index idx) const noexcept -> const Contained& {
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
enum class Layout : uint8_t { C, F };
template <typename Contained, Index M, Index N, Index NGHOST = 0, Layout LAYOUT = Layout::C>
requires(M > 0 && N > 0 && NGHOST >= 0)
class Matrix {
  static constexpr auto ARRAY_SIZE = (M + 2 * NGHOST) * (N + 2 * NGHOST);

 public:
  static constexpr bool is_small = ARRAY_SIZE * sizeof(Contained) < 1024UZ;

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

  constexpr auto operator[](Index i, Index j) noexcept -> Contained& {
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

  constexpr auto operator[](Index i, Index j) const noexcept -> const Contained& {
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
      *c.get_data(),
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
enum class ExecutionPolicy : uint8_t { SERIAL, PARALLEL };

template <typename FUNC>
concept ForEachFunc = requires(FUNC f) {
  { f(std::declval<Index>(), std::declval<Index>()) } -> std::same_as<void>;
};

template <ExecutionPolicy EXEC = ExecutionPolicy::SERIAL, ForEachFunc FUNC>
inline void for_each(Index i_min, Index i_max, Index j_min, Index j_max, FUNC&& f) noexcept {
  if constexpr (EXEC == ExecutionPolicy::SERIAL) {
    for (Index i = i_min; i < i_max; ++i) {
      for (Index j = j_min; j < j_max; ++j) {
        f(i, j);
      }
    }
  } else {
#pragma omp parallel for collapse(2)
    for (Index i = i_min; i < i_max; ++i) {
      for (Index j = j_min; j < j_max; ++j) {
        f(i, j);
      }
    }
  }
}

template <ExecutionPolicy EXEC = ExecutionPolicy::SERIAL,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          ForEachFunc FUNC>
inline void for_each_i(const Matrix<Float, NX, NY, NGHOST>& _, FUNC&& f) noexcept {
  for_each<EXEC>(0, NX, 0, NY, std::forward<FUNC&&>(f));
}

template <ExecutionPolicy EXEC = ExecutionPolicy::SERIAL,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          ForEachFunc FUNC>
inline void for_each_a(const Matrix<Float, NX, NY, NGHOST>& _, FUNC&& f) noexcept {
  for_each<EXEC>(-NGHOST, NX + NGHOST, -NGHOST, NY + NGHOST, std::forward<FUNC&&>(f));
}

#endif  // FLUID_SOLVER_CONTAINER_HPP_
