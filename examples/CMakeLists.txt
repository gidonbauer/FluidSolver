set(executables
  IncompSolver
  TwoPhaseSolver
  RisingBubble
  WallBubble
  Wave

  II
  DiffuseIB
  SharpIB
  MovingLeastSquaresIB
  DiffuseIB-DFG
  SharpIB-DFG
  VerifyDFGBenchmarkSetup

  VOF
  Curvature

  GrowingIB
  DivFreeExtrapolation
)

foreach(exec ${executables})
    # - Define executables -------------
    add_executable(${exec} ${exec}.cpp)

    # - Define include path ------------
    target_include_directories(${exec} PRIVATE ${CMAKE_SOURCE_DIR}/src/)

    # - Link libraries -----------------
    if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "NVHPC")
      target_link_libraries(${exec} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(${exec} PRIVATE Igor)
    target_link_libraries(${exec} PRIVATE irl)
    target_link_libraries(${exec} PRIVATE HYPRE::HYPRE)
    if (HDF5_FOUND)
      target_link_libraries(${exec} PRIVATE HDF5::HDF5 hdf5::hdf5_hl hdf5::hdf5_hl_cpp)
    endif()
endforeach()


