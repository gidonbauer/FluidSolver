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
using Index = std::int64_t;
#else
static_assert(std::is_integral_v<FS_INDEX_TYPE> && std::is_signed_v<FS_INDEX_TYPE>,
              "FS_INDEX_TYPE must be a signed integer type.");
using Index = FS_INDEX_TYPE;
#endif  // FS_INDEX_TYPE

enum class Layout : uint8_t { C, F };

// =================================================================================================
template <typename Contained, Index N, Index NGHOST = 0>
requires(N > 0 && NGHOST >= 0)
class Field1D {
  static constexpr auto ARRAY_SIZE = static_cast<size_t>(N + 2 * NGHOST);

 private:
  std::unique_ptr<std::array<Contained, ARRAY_SIZE>> m_data{};

  [[nodiscard]] constexpr auto get_idx(Index raw_idx) const noexcept -> Index {
    return raw_idx + NGHOST;
  }

 public:
  Field1D() noexcept
      : m_data(new std::array<Contained, ARRAY_SIZE>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data->data(), size(), Contained{0});
    }
  }
  Field1D(const Field1D& other) noexcept                              = delete;
  Field1D(Field1D&& other) noexcept                                   = delete;
  constexpr auto operator=(const Field1D& other) noexcept -> Field1D& = delete;
  constexpr auto operator=(Field1D&& other) noexcept -> Field1D&      = delete;
  ~Field1D() noexcept                                                 = default;

  [[nodiscard]] constexpr auto operator()(Index idx) noexcept -> Contained& {
    IGOR_ASSERT(idx >= -NGHOST && idx < N + NGHOST,
                "Index {} is out of bounds for Field1D with dimension {}:{}",
                idx,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(idx));
  }

  [[nodiscard]] constexpr auto operator()(Index idx) const noexcept -> const Contained& {
    IGOR_ASSERT(idx >= -NGHOST && idx < N + NGHOST,
                "Index {} is out of bounds for Field1D with dimension {}:{}",
                idx,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(idx));
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return m_data->data(); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return m_data->data();
  }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return ARRAY_SIZE; }

  [[nodiscard]] constexpr auto n_ghost() const noexcept -> Index { return NGHOST; }

  [[nodiscard]] constexpr auto extent([[maybe_unused]] Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 1, "Dimension {} is out of bounds for Field1D", r);
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
class Field2D {
  static constexpr auto ARRAY_SIZE = (M + 2 * NGHOST) * (N + 2 * NGHOST);

 private:
  std::unique_ptr<std::array<Contained, ARRAY_SIZE>> m_data{};

 public:
  [[nodiscard]] constexpr auto get_idx(Index i, Index j) const noexcept -> Index {
    if constexpr (LAYOUT == Layout::C) {
      return (j + NGHOST) + (i + NGHOST) * (N + 2 * NGHOST);
    } else {
      return (i + NGHOST) + (j + NGHOST) * (M + 2 * NGHOST);
    }
  }

  Field2D() noexcept
      : m_data(new std::array<Contained, ARRAY_SIZE>) {
    IGOR_ASSERT(m_data != nullptr, "Allocation failed.");
    if constexpr (std::is_arithmetic_v<Contained>) {
      std::fill_n(m_data->data(), size(), Contained{0});
    }
  }
  Field2D(const Field2D& other) noexcept                              = delete;
  Field2D(Field2D&& other) noexcept                                   = delete;
  constexpr auto operator=(const Field2D& other) noexcept -> Field2D& = delete;
  constexpr auto operator=(Field2D&& other) noexcept -> Field2D&      = delete;
  ~Field2D() noexcept                                                 = default;

  constexpr auto operator()(Index i, Index j) noexcept -> Contained& {
    IGOR_ASSERT(i >= -NGHOST && i < M + NGHOST && j >= -NGHOST && j < N + NGHOST,
                "Index ({}, {}) is out of bounds for Field2D of size {}:{}x{}:{}",
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
                "Index ({}, {}) is out of bounds for Field2D of size {}:{}x{}:{}",
                i,
                j,
                -NGHOST,
                M + NGHOST,
                -NGHOST,
                N + NGHOST);
    return *(get_data() + get_idx(i, j));
  }

  [[nodiscard]] constexpr auto get_data() noexcept -> Contained* { return m_data->data(); }
  [[nodiscard]] constexpr auto get_data() const noexcept -> const Contained* {
    return m_data->data();
  }

  [[nodiscard]] constexpr auto size() const noexcept -> Index { return ARRAY_SIZE; }
  [[nodiscard]] constexpr auto n_ghost() const noexcept -> Index { return NGHOST; }

  [[nodiscard]] constexpr auto extent(Index r) const noexcept -> Index {
    IGOR_ASSERT(r >= 0 && r < 2, "Dimension {} is out of bounds for Field1D", r);
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
