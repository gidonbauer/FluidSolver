cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

option(FS_BUILD_TEST "Build tests" ON)
option(FS_BUILD_BENCH "Build benchmarks" OFF)
option(FS_FAST "Enable unsafe optimizations." OFF)
option(FS_USE_SCOREP "Enable profiling using Score-P." OFF)

include(cmake/compiler_flags.cmake)
add_compile_definitions(FS_BASE_DIR="${CMAKE_SOURCE_DIR}")

find_package(OpenMP REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL)
if (NOT HDF5_FOUND)
  add_compile_definitions(FS_DISABLE_HDF)
endif()

# = Add dependencies ===============================================================================
# = Igor =========================================
add_subdirectory(${CMAKE_SOURCE_DIR}/Submodules/Igor)

# = IRL ==========================================
set(IRL_BUILD_FORTRAN OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/Submodules/interface-reconstruction-library SYSTEM)

# = Hypre ========================================
set(HYPRE_ENABLE_MPI          OFF CACHE INTERNAL "")
set(HYPRE_ENABLE_OPENMP       ON  CACHE INTERNAL "")
# set(HYPRE_ENABLE_PRINT_ERRORS ON  CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/Submodules/hypre/src SYSTEM)
# = Add dependencies ===============================================================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${CMAKE_SOURCE_DIR}/examples/)

if (FS_BUILD_TEST)
  enable_testing()
  add_subdirectory(${CMAKE_SOURCE_DIR}/test/)
endif()

if (FS_BUILD_BENCH)
  add_subdirectory(${CMAKE_SOURCE_DIR}/bench/)
endif()
