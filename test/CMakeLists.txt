set(tests
  # LaminarChannel_0
  # LaminarChannel_1
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

enable_testing()

foreach(exec ${tests})
    # - Define executables -------------
    if (${exec} STREQUAL "LaminarChannel_0")
      add_executable(${exec} LaminarChannel.cpp)
      target_compile_definitions(${exec}, LC_U_INIT=0)
    elseif (${exec} STREQUAL "LaminarChannel_1")
      add_executable(${exec} LaminarChannel.cpp)
      target_compile_definitions(${exec}, LC_U_INIT=1)
    else()
      add_executable(${exec} ${exec}.cpp)
    endif()

    # - Define include path ------------
    target_include_directories(${exec} PRIVATE ${CMAKE_SOURCE_DIR}/src/)

    # - Link libraries -----------------
    target_link_libraries(${exec} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${exec} PRIVATE Igor)
    target_link_libraries(${exec} PRIVATE irl)
    target_link_libraries(${exec} PRIVATE HYPRE::HYPRE)
    if (HDF5_FOUND)
      target_link_libraries(${exec} PRIVATE HDF5::HDF5 hdf5::hdf5_hl hdf5::hdf5_hl_cpp)
    endif()

    add_test(NAME ${exec} COMMAND ${exec})
endforeach()


