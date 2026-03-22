#ifndef FLUID_SOLVER_MACROS_HPP_
#define FLUID_SOLVER_MACROS_HPP_

#include <Igor/Macros.hpp>

#if defined(__GNUC__)
#define FS_COMP_GNUC 1
#else
#define FS_COMP_GNUC 0
#endif

#if defined(__clang__)
#define FS_COMP_CLANG 1
#else
#define FS_COMP_CLANG 0
#endif

#if defined(__llvm__)
#define FS_COMP_LLVM 1
#else
#define FS_COMP_LLVM 0
#endif

#if defined(__INTEL_COMPILER)
#define FS_COMP_INTEL 1
#else
#define FS_COMP_INTEL 0
#endif

#if FS_COMP_GNUC
#define FS_ALWAYS_INLINE __attribute__((flatten)) inline __attribute__((always_inline))
#else
#define FS_ALWAYS_INLINE inline
#endif

// =================================================================================================

// #define FS_PARALLEL
#ifdef FS_PARALLEL

#define FS_PARALLEL_CONSTEXPR inline
#define FS_PARALLEL_FOR(...) _Pragma(IGOR_XSTRINGIFY(omp parallel for __VA_ARGS__))

#else

#define FS_PARALLEL_CONSTEXPR constexpr
#define FS_PARALLEL_FOR(...)

#endif

#define FS_FOR_EACH_I_1D(f, idx) for (Index idx = 0; idx < (f).extent(0); ++idx)
#define FS_FOR_EACH_A_1D(f, idx)                                                                   \
  for (Index idx = -(f).nghost(); idx < (f).extent(0) + (f).nghost(); ++idx)

#define FS_FOR_EACH_I(f)                                                                           \
  for (Index i = 0; i < (f).extent(0); ++i)                                                        \
    for (Index j = 0; j < (f).extent(1); ++j)

#define FS_FOR_EACH_A(f)                                                                           \
  for (Index i = -(f).nghost(); i < (f).extent(0) + (f).nghost(); ++i)                             \
    for (Index j = -(f).nghost(); j < (f).extent(1) + (f).nghost(); ++j)

#endif  // FLUID_SOLVER_MACROS_HPP_
