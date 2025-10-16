set(tests
  LaminarChannel_0
  LaminarChannel_1
  Couette
  TaylorGreenVortex
  ConstantVelocityVOF
  LinearVelocityVOF
  TaylorGreenVortexVOF
  MovingDrop
  StationaryDrop
  Operators
  Container
  Utility
  OffloadGPU
)

foreach(test ${tests})
    # - Define executables -------------
    if (${test} STREQUAL "LaminarChannel_0")
      add_executable(${test} LaminarChannel.cpp)
      target_compile_definitions(${test} PRIVATE LC_U_INIT=0)
    elseif (${test} STREQUAL "LaminarChannel_1")
      add_executable(${test} LaminarChannel.cpp)
      target_compile_definitions(${test} PRIVATE LC_U_INIT=1)
    else()
      add_executable(${test} ${test}.cpp)
    endif()

    # - Define include path ------------
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/src/)

    # - Link libraries -----------------
    if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "NVHPC")
      target_link_libraries(${exec} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(${test} PRIVATE Igor)
    target_link_libraries(${test} PRIVATE irl)
    target_link_libraries(${test} PRIVATE HYPRE::HYPRE)
    if (HDF5_FOUND)
      target_link_libraries(${test} PRIVATE HDF5::HDF5 hdf5::hdf5_hl hdf5::hdf5_hl_cpp)
    endif()

    add_test(NAME ${test} COMMAND ${test})
endforeach()
