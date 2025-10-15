cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/compiler_flags.cmake)

find_package(OpenMP REQUIRED)

find_package(HDF5 COMPONENTS C CXX HL)
if (NOT HDF5_FOUND)
  add_compile_definitions(FS_DISABLE_HDF)
endif()

# = Fetch dependencies =====================================
include(FetchContent)

# = Igor =========================================
FetchContent_Declare(
  Igor
  GIT_REPOSITORY https://github.com/gidonbauer/Igor
  GIT_TAG        main
)
FetchContent_MakeAvailable(Igor)

# = IRL ==========================================
FetchContent_Declare(
  IRL
  GIT_REPOSITORY https://github.com/robert-chiodi/interface-reconstruction-library
  GIT_TAG        master
  SYSTEM
)
set(IRL_BUILD_FORTRAN OFF CACHE INTERNAL "")
set(IRL_USE_ABSL      OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(IRL)

# = Hypre ========================================
FetchContent_Declare(
  HYPRE
  GIT_REPOSITORY https://github.com/hypre-space/hypre
  GIT_TAG        master
  SOURCE_SUBDIR  src
)
set(HYPRE_ENABLE_MPI          OFF CACHE INTERNAL "")
set(HYPRE_ENABLE_OPENMP       ON  CACHE INTERNAL "")
set(HYPRE_ENABLE_PRINT_ERRORS ON  CACHE INTERNAL "")
FetchContent_MakeAvailable(HYPRE)
# = Fetch dependencies =====================================

add_subdirectory(${CMAKE_SOURCE_DIR}/examples/)
