#ifndef FLUID_SOLVER_MACROS_HPP_
#define FLUID_SOLVER_MACROS_HPP_

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

#endif  // FLUID_SOLVER_MACROS_HPP_
