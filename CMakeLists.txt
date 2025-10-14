cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DEFAULT_BUILD_TYPE "Release")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS         "-Wall -Wextra -pedantic -Wshadow -Wconversion -Winline -std=c++23")
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  set(CMAKE_CXX_FLAGS_FAST    "-march=native -O3 -ffast-math -DIGOR_NDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS         "-Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23")
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3")
  set(CMAKE_CXX_FLAGS_FAST    "-march=native -O3 -ffast-math -DIGOR_NDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
  set(CMAKE_CXX_FLAGS         "-Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model precise")
  set(CMAKE_CXX_FLAGS_FAST    "-O3 -xSSE4.2 -axCORE-AVX2,AVX -fp-model fast=2 -ffast-math -DIGOR_NDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "NVHPC")
  set(CMAKE_CXX_FLAGS         "-Wall -Wextra -pedantic -Wshadow -std=c++23")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fastsse -Mvect=simd:256,noassoc")
  set(CMAKE_CXX_FLAGS_FAST    "-O3 -fast -fastsse -Mvect=simd:256")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
  # CXX_SANITIZER_FLAGS = -fsanitize=address,leak,undefined
else()
  message(FATAL_ERROR "Unknown C++ compiler `${CMAKE_CXX_COMPILER}` with ID `${CMAKE_CXX_COMPILER_ID}`")
endif()

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
