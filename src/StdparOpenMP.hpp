#ifndef FLUID_SOLVER_STDPAR_OPENMP_HPP_
#define FLUID_SOLVER_STDPAR_OPENMP_HPP_

#include <cstdint>
#include <type_traits>

enum class StdparOpenMP : uint8_t { Serial, Parallel, ParallelDynamic };

namespace std {

template <class ExecPolicy, class RandIt, class UnaryFunction>
requires(std::is_same_v<ExecPolicy, StdparOpenMP>)
void for_each(ExecPolicy policy, RandIt first, RandIt last, UnaryFunction&& f) {
  switch (policy) {
    case StdparOpenMP::Serial:
      for (auto iter = first; iter != last; ++iter) {
        f(*iter);
      }
      break;
    case StdparOpenMP::Parallel:
#pragma omp parallel for
      for (auto iter = first; iter != last; ++iter) {
        f(*iter);
      }
      break;
    case StdparOpenMP::ParallelDynamic:
#pragma omp parallel for schedule(dynamic)
      for (auto iter = first; iter != last; ++iter) {
        f(*iter);
      }
      break;
  }
}

}  // namespace std

#endif  // FLUID_SOLVER_STDPAR_OPENMP_HPP_
