set(executables
  dmomdt-no-fuse
  dmomdt-fuse-flux
  dmomdt-fuse-all
  update-velo-fuse
  LinearSolver
)

foreach(exec ${executables})
    # - Define executables -------------
    if (${exec} STREQUAL "dmomdt-no-fuse")
      add_executable(${exec} dmomdt.cpp)
    elseif(${exec} STREQUAL "dmomdt-fuse-flux")
      add_executable(${exec} dmomdt.cpp)
      target_compile_definitions(${exec} PRIVATE FS_FUSE_MOM_FLUX)
    elseif(${exec} STREQUAL "dmomdt-fuse-all")
      add_executable(${exec} dmomdt.cpp)
      target_compile_definitions(${exec} PRIVATE FS_FUSE_MOM_ALL)
    else()
      add_executable(${exec} ${exec}.cpp)
    endif()

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

    if (${exec} STREQUAL "LinearSolver")
      target_compile_definitions(${exec} PRIVATE "ACCELERATE_NEW_LAPACK")
      target_link_libraries(${exec} PRIVATE "-framework Accelerate")
    endif()
endforeach()
