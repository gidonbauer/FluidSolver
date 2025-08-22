#ifndef FLUID_SOLVER_FOR_EACH_HPP_
#define FLUID_SOLVER_FOR_EACH_HPP_

#include <algorithm>

#include "Container.hpp"
#include "IotaIter.hpp"
#include "Macros.hpp"
#include "StdparOpenMP.hpp"

#ifndef FS_PARALLEL_THRESHOLD
constexpr Index FS_PARALLEL_THRESHOLD_COUNT = 1000;
#else
static_assert(std::is_convertible_v<std::remove_cvref_t<decltype(FS_PARALLEL_THRESHOLD)>, Index>,
              "FS_PARALLEL_THRESHOLD must have a value that must be convertible to Index.");
constexpr Index FS_PARALLEL_THRESHOLD_COUNT = FS_PARALLEL_THRESHOLD;
#endif  // FS_PARALLEL_THRESHOLD

// -------------------------------------------------------------------------------------------------
enum class Exec : uint8_t { Serial, Parallel, ParallelDynamic };

// -------------------------------------------------------------------------------------------------
template <typename FUNC>
concept ForEachFunc1D = requires(FUNC f) {
  { f(std::declval<Index>()) } -> std::same_as<void>;
};

// -------------------------------------------------------------------------------------------------
template <Index I_MIN, Index I_MAX, Exec EXEC = Exec::Serial, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each(FUNC&& f) noexcept {
  constexpr StdparOpenMP policy = []() {
    if constexpr (EXEC == Exec::Serial || (I_MAX - I_MIN) < FS_PARALLEL_THRESHOLD_COUNT) {
      return StdparOpenMP::Serial;
    } else if constexpr (EXEC == Exec::Parallel) {
      return StdparOpenMP::Parallel;
    } else {
      return StdparOpenMP::ParallelDynamic;
    }
  }();
  std::for_each(policy, IotaIter<Index>(I_MIN), IotaIter<Index>(I_MAX), std::forward<FUNC&&>(f));
}

// -------------------------------------------------------------------------------------------------
template <Exec EXEC = Exec::Serial, typename Float, Index N, Index NGHOST, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each_i(const Vector<Float, N, NGHOST>& _, FUNC&& f) noexcept {
  for_each<0, N, EXEC>(std::forward<FUNC&&>(f));
}

// -------------------------------------------------------------------------------------------------
template <Exec EXEC = Exec::Serial, typename Float, Index N, Index NGHOST, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each_a(const Vector<Float, N, NGHOST>& _, FUNC&& f) noexcept {
  for_each<-NGHOST, N + NGHOST, EXEC>(std::forward<FUNC&&>(f));
}

// -------------------------------------------------------------------------------------------------
template <typename FUNC>
concept ForEachFunc2D = requires(FUNC f) {
  { f(std::declval<Index>(), std::declval<Index>()) } -> std::same_as<void>;
};

// -------------------------------------------------------------------------------------------------
template <Index I_MIN,
          Index I_MAX,
          Index J_MIN,
          Index J_MAX,
          Exec EXEC     = Exec::Serial,
          Layout LAYOUT = Layout::C,
          ForEachFunc2D FUNC>
FS_ALWAYS_INLINE void for_each(FUNC&& f) noexcept {
  constexpr StdparOpenMP policy = []() {
    if constexpr (EXEC == Exec::Serial ||
                  (I_MAX - I_MIN) * (J_MAX - J_MIN) < FS_PARALLEL_THRESHOLD_COUNT) {
      return StdparOpenMP::Serial;
    } else if constexpr (EXEC == Exec::Parallel) {
      return StdparOpenMP::Parallel;
    } else {
      return StdparOpenMP::ParallelDynamic;
    }
  }();

  constexpr auto from_linear_index = [](Index idx) noexcept -> std::pair<Index, Index> {
    if constexpr (LAYOUT == Layout::C) {
      return {
          idx / (J_MAX - J_MIN) + I_MIN,
          idx % (J_MAX - J_MIN) + J_MIN,
      };
    } else {
      return {
          idx % (I_MAX - I_MIN) + I_MIN,
          idx / (I_MAX - I_MIN) + J_MIN,
      };
    }
  };

  std::for_each(policy,
                IotaIter<Index>(0),
                IotaIter<Index>((I_MAX - I_MIN) * (J_MAX - J_MIN)),
                [&](Index idx) {
                  const auto [i, j] = from_linear_index(idx);
                  f(i, j);
                });
}

// -------------------------------------------------------------------------------------------------
template <Exec EXEC = Exec::Serial,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          Layout LAYOUT,
          ForEachFunc2D FUNC>
FS_ALWAYS_INLINE void for_each_i(const Matrix<Float, NX, NY, NGHOST, LAYOUT>& _,
                                 FUNC&& f) noexcept {
  for_each<0, NX, 0, NY, EXEC, LAYOUT>(std::forward<FUNC&&>(f));
}

// -------------------------------------------------------------------------------------------------
template <Exec EXEC = Exec::Serial,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          Layout LAYOUT,
          ForEachFunc2D FUNC>
FS_ALWAYS_INLINE void for_each_a(const Matrix<Float, NX, NY, NGHOST, LAYOUT>& _,
                                 FUNC&& f) noexcept {
  for_each<-NGHOST, NX + NGHOST, -NGHOST, NY + NGHOST, EXEC, LAYOUT>(std::forward<FUNC&&>(f));
}

#endif  // FLUID_SOLVER_FOR_EACH_HPP_
