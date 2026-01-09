set(executables
  PeriodicChannel
)

foreach(N  RANGE 21 101 10)
    # - Define executables -------------
    set(exec PeriodicChannel-${N})
    add_executable(${exec} PeriodicChannel.cpp)
    target_compile_definitions(${exec} PRIVATE FS_SCALING_N=${N})

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

add_custom_target(Run-PeriodicChannel-Scaling
  COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/scaling/output
  COMMAND echo "NX,NY,dx,dy,pressure_error,dpdx_error,U_error,V_error,runtime_s" > ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-21  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-31  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-41  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-51  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-61  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-71  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-81  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-91  --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  COMMAND PeriodicChannel-101 --csv >> ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
  BYPRODUCTS ${CMAKE_SOURCE_DIR}/scaling/output/PeriodicChannel-Scaling.csv
)
