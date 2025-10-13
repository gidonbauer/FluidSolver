cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(FS_DISABLE_HDF)

find_package(OpenMP REQUIRED)

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
set(IRL_BUILD_FORTRAN ON)
set(IRL_USE_ABSL OFF)
FetchContent_Declare(
  IRL
  GIT_REPOSITORY https://github.com/robert-chiodi/interface-reconstruction-library
  GIT_TAG        master
)
FetchContent_MakeAvailable(IRL)

# = Hypre ========================================
set(HYPRE_ENABLE_MPI OFF)
set(HYPRE_ENABLE_OPENMP ON)
FetchContent_Declare(
  HYPRE
  GIT_REPOSITORY https://github.com/hypre-space/hypre
  GIT_TAG        master
  SOURCE_SUBDIR  src
)
FetchContent_MakeAvailable(HYPRE)
# = Fetch dependencies =====================================

add_subdirectory(${CMAKE_SOURCE_DIR}/examples/)
