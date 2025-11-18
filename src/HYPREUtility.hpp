#ifndef FLUID_SOLVER_HYPRE_SETUP_HPP_
#define FLUID_SOLVER_HYPRE_SETUP_HPP_

#include <omp.h>

#include <HYPRE_utilities.h>

#include "Container.hpp"

#ifndef FS_HYPRE_PARALLEL_THRESHOLD
#ifdef __APPLE__
constexpr Index FS_HYPRE_PARALLEL_GRID_SIZE_THRESHOLD = 1000 * 1000;
#else
constexpr Index FS_HYPRE_PARALLEL_GRID_SIZE_THRESHOLD = 500 * 500;
#endif
#else
static_assert(
    std::is_convertible_v<std::remove_cvref_t<decltype(FS_HYPRE_PARALLEL_THRESHOLD)>, Index>,
    "PS_PARALLEL_THRESHOLD must have a value that must be convertible to Index.");
constexpr Index FS_HYPRE_PARALLEL_GRID_SIZE_THRESHOLD = FS_HYPRE_PARALLEL_THRESHOLD;
#endif  // FS_HYPRE_PARALLEL_THRESHOLD

#define FS_HYPRE_MAKE_SINGLE_THREADED_IF_NECESSARY                                                 \
  int prev_num_threads = -1;                                                                       \
  if constexpr ((NX + 2 * NGHOST) * (NY + 2 * NGHOST) < FS_HYPRE_PARALLEL_GRID_SIZE_THRESHOLD) {   \
    _Pragma("omp parallel") _Pragma("omp single") { prev_num_threads = omp_get_num_threads(); }    \
    omp_set_num_threads(1);                                                                        \
  }

#define FS_HYPRE_RESET_THREADING                                                                   \
  if constexpr ((NX + 2 * NGHOST) * (NY + 2 * NGHOST) < FS_HYPRE_PARALLEL_GRID_SIZE_THRESHOLD) {   \
    omp_set_num_threads(prev_num_threads);                                                         \
  }

enum class HypreSolver : std::uint8_t { GMRES, PCG, BiCGSTAB, SMG, PFMG };
enum class HyprePrecond : std::uint8_t { SMG, PFMG, NONE };

namespace detail {

int hypre_use_count = 0;  // NOLINT

void initialize_hypre() {
  if (hypre_use_count == 0) { HYPRE_Initialize(); }
  hypre_use_count += 1;
}

void finalize_hypre() {
  hypre_use_count -= 1;
  if (hypre_use_count == 0) { HYPRE_Finalize(); }
}

}  // namespace detail

#endif  // FLUID_SOLVER_HYPRE_SETUP_HPP_
