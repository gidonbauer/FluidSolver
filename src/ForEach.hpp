#ifndef FLUID_SOLVER_FOR_EACH_HPP_
#define FLUID_SOLVER_FOR_EACH_HPP_

#include "Container.hpp"

#ifndef FS_PARALLEL_THRESHOLD
constexpr Index FS_PARALLEL_THRESHOLD_COUNT = 1000;
#else
static_assert(std::is_convertible_v<std::remove_cvref_t<decltype(FS_PARALLEL_THRESHOLD)>, Index>,
              "FS_PARALLEL_THRESHOLD must have a value that must be convertible to Index.");
constexpr Index FS_PARALLEL_THRESHOLD_COUNT = FS_PARALLEL_THRESHOLD;
#endif  // FS_PARALLEL_THRESHOLD

// -------------------------------------------------------------------------------------------------
enum class Exec : uint8_t { Serial, Parallel };

// -------------------------------------------------------------------------------------------------
template <typename FUNC>
concept ForEachFunc1D = requires(FUNC f) {
  { f(std::declval<Index>()) } -> std::same_as<void>;
};

// -------------------------------------------------------------------------------------------------
template <Index I_MIN, Index I_MAX, Exec EXEC = Exec::Serial, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each(FUNC&& f) noexcept {
  if constexpr (EXEC == Exec::Serial) {
    for (Index i = I_MIN; i < I_MAX; ++i) {
      f(i);
    }
  } else {
#pragma omp parallel for if ((I_MAX - I_MIN) > FS_PARALLEL_THRESHOLD_COUNT)
    for (Index i = I_MIN; i < I_MAX; ++i) {
      f(i);
    }
  }
}

template <Exec EXEC = Exec::Serial, typename Float, Index N, Index NGHOST, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each_i(const Vector<Float, N, NGHOST>& _, FUNC&& f) noexcept {
  for_each<0, N, EXEC>(std::forward<FUNC&&>(f));
}

template <Exec EXEC = Exec::Serial, typename Float, Index N, Index NGHOST, ForEachFunc1D FUNC>
FS_ALWAYS_INLINE void for_each_a(const Vector<Float, N, NGHOST>& _, FUNC&& f) noexcept {
  for_each<-NGHOST, N + NGHOST, EXEC>(std::forward<FUNC&&>(f));
}

// -------------------------------------------------------------------------------------------------
template <typename FUNC>
concept ForEachFunc2D = requires(FUNC f) {
  { f(std::declval<Index>(), std::declval<Index>()) } -> std::same_as<void>;
};

template <Index I_MIN,
          Index I_MAX,
          Index J_MIN,
          Index J_MAX,
          Exec EXEC     = Exec::Serial,
          Layout LAYOUT = Layout::C,
          ForEachFunc2D FUNC>
FS_ALWAYS_INLINE void for_each(FUNC&& f) noexcept {
  if constexpr (EXEC == Exec::Serial && LAYOUT == Layout::C) {

    for (Index i = I_MIN; i < I_MAX; ++i) {
      for (Index j = J_MIN; j < J_MAX; ++j) {
        f(i, j);
      }
    }

  } else if constexpr (EXEC == Exec::Parallel && LAYOUT == Layout::C) {

#pragma omp parallel for collapse(2) if ((I_MAX - I_MIN) * (J_MAX - J_MIN) >                       \
                                             FS_PARALLEL_THRESHOLD_COUNT)
    for (Index i = I_MIN; i < I_MAX; ++i) {
      for (Index j = J_MIN; j < J_MAX; ++j) {
        f(i, j);
      }
    }

  } else if constexpr (EXEC == Exec::Serial && LAYOUT == Layout::F) {

    for (Index j = J_MIN; j < J_MAX; ++j) {
      for (Index i = I_MIN; i < I_MAX; ++i) {
        f(i, j);
      }
    }

  } else if constexpr (EXEC == Exec::Parallel && LAYOUT == Layout::F) {

#pragma omp parallel for collapse(2) if ((I_MAX - I_MIN) * (J_MAX - J_MIN) >                       \
                                             FS_PARALLEL_THRESHOLD_COUNT)
    for (Index j = J_MIN; j < J_MAX; ++j) {
      for (Index i = I_MIN; i < I_MAX; ++i) {
        f(i, j);
      }
    }

  } else {
    Igor::Panic(
        "Unreachable: EXEC={}, LAYOUT={}", static_cast<int>(EXEC), static_cast<int>(LAYOUT));
    std::unreachable();
  }
}

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
