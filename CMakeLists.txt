cmake_minimum_required(VERSION 3.26)

project(FluidSolver LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Igor
  GIT_REPOSITORY https://github.com/gidonbauer/Igor
  GIT_TAG        main
)
FetchContent_MakeAvailable(Igor)


FetchContent_Declare(
  Irl
  GIT_REPOSITORY https://github.com/robert-chiodi/interface-reconstruction-library
  GIT_TAG        master
)
FetchContent_MakeAvailable(Irl)


add_subdirectory(${CMAKE_SOURCE_DIR}/examples/)
