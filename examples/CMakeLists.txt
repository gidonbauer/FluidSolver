set(executables
  IncompSolver
  VOF
  Curvature
  TwoPhaseSolver
  IB
  PhaseChange
  RisingBubble
)

foreach(exec ${executables})
    # - Define executables -------------
    add_executable(${exec} ${exec}.cpp)

    # - Define include path ------------
    target_include_directories(${exec} PRIVATE ${CMAKE_SOURCE_DIR}/src/)

    # - Link libraries -----------------
    target_link_libraries(${exec} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${exec} PRIVATE Igor)
    target_link_libraries(${exec} PRIVATE irl)
    target_link_libraries(${exec} PRIVATE HYPRE::HYPRE)
    if (HDF5_FOUND)
      target_link_libraries(${exec} PRIVATE HDF5)
    endif()
endforeach()


