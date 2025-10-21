#ifndef FLUID_SOLVER_IOTA_ITER_HPP_
#define FLUID_SOLVER_IOTA_ITER_HPP_

#include <concepts>
#include <iterator>

// Adapted from poolSTL: https://github.com/alugowski/poolSTL
template <std::signed_integral Integer>
class IotaIter {
  Integer m_value = 0;

 public:
  using value_type              = Integer;
  using difference_type         = Integer;
  using pointer                 = Integer*;
  using reference               = Integer;
  using iterator_category       = std::random_access_iterator_tag;

  constexpr IotaIter() noexcept = default;
  explicit IotaIter(Integer value) noexcept
      : m_value(value) {}
  constexpr auto operator=(Integer rhs) noexcept -> IotaIter<Integer>& {
    m_value = rhs;
    return *this;
  }
  constexpr IotaIter(const IotaIter<Integer>& other) noexcept                  = default;
  constexpr IotaIter(IotaIter<Integer>&& other) noexcept                       = default;
  constexpr auto operator=(const IotaIter& rhs) noexcept -> IotaIter<Integer>& = default;
  constexpr auto operator=(IotaIter&& rhs) noexcept -> IotaIter<Integer>&      = default;
  constexpr ~IotaIter() noexcept                                               = default;

  constexpr auto operator*() const noexcept -> Integer { return m_value; }
  constexpr auto operator[](difference_type rhs) const noexcept -> Integer { return m_value + rhs; }
  // operator-> has no meaning in this application

  constexpr auto operator==(const IotaIter<Integer>& rhs) const noexcept -> bool = default;
  constexpr auto operator<=>(const IotaIter<Integer>& rhs) const noexcept        = default;

  constexpr auto operator+=(difference_type rhs) noexcept -> IotaIter<Integer>& {
    m_value += rhs;
    return *this;
  }
  constexpr auto operator-=(difference_type rhs) noexcept -> IotaIter<Integer>& {
    m_value -= rhs;
    return *this;
  }

  constexpr auto operator++() noexcept -> IotaIter<Integer>& {
    ++m_value;
    return *this;
  }
  constexpr auto operator--() noexcept -> IotaIter<Integer>& {
    --m_value;
    return *this;
  }
  constexpr auto operator++(int) noexcept -> IotaIter<Integer> {
    IotaIter<Integer> ret(m_value);
    ++m_value;
    return ret;
  }
  constexpr auto operator--(int) noexcept -> IotaIter<Integer> {
    IotaIter<Integer> ret(m_value);
    --m_value;
    return ret;
  }

  constexpr auto operator-(const IotaIter<Integer>& rhs) const noexcept -> difference_type {
    return m_value - rhs.m_value;
  }
  constexpr auto operator-(difference_type rhs) const noexcept -> IotaIter<Integer> {
    return IotaIter(m_value - rhs);
  }
  constexpr auto operator+(difference_type rhs) const noexcept -> IotaIter<Integer> {
    return IotaIter(m_value + rhs);
  }

  friend constexpr auto operator+(difference_type lhs, const IotaIter<Integer>& rhs) noexcept
      -> IotaIter<Integer> {
    return IotaIter(lhs + rhs.m_value);
  }
};

#endif  // FLUID_SOLVER_IOTA_ITER_HPP_
