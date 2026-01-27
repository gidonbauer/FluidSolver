# = PeriodicChannel ================================================================================
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

# = Lid driven flow ================================================================================
foreach(N  RANGE 4 10 1)
    # - Define executables -------------
    set(exec LidDrivenFlow-${N})
    add_executable(${exec} LidDrivenFlow.cpp)
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

add_custom_target(Run-LidDrivenFlow-Scaling
  COMMAND LidDrivenFlow-4
  COMMAND LidDrivenFlow-5
  COMMAND LidDrivenFlow-6
  COMMAND LidDrivenFlow-7
  COMMAND LidDrivenFlow-8
  COMMAND LidDrivenFlow-9
  COMMAND LidDrivenFlow-10
)

# = Diffuse Immersed Boundaries - Channel flow =====================================================
foreach(N  RANGE 4 8 1)
    # - Define executables -------------
    set(exec DiffuseIBChannel-${N})
    add_executable(${exec} DiffuseIBChannel.cpp)
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

add_custom_target(Run-DiffuseIBChannel-Scaling
  COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/scaling/output
  COMMAND DiffuseIBChannel-4 --write-csv --write-csv-header > ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
  COMMAND DiffuseIBChannel-5 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
  COMMAND DiffuseIBChannel-6 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
  COMMAND DiffuseIBChannel-7 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
  COMMAND DiffuseIBChannel-8 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
  BYPRODUCTS ${CMAKE_SOURCE_DIR}/scaling/output/DiffuseIBChannel-Scaling.csv
)

# = Sharp Immersed Boundaries - Channel flow =======================================================
foreach(N  RANGE 4 8 1)
    # - Define executables -------------
    set(exec SharpIBChannel-${N})
    add_executable(${exec} SharpIBChannel.cpp)
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

add_custom_target(Run-SharpIBChannel-Scaling
  COMMAND mkdir -p ${CMAKE_SOURCE_DIR}/scaling/output
  COMMAND SharpIBChannel-4 --write-csv --write-csv-header > ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
  COMMAND SharpIBChannel-5 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
  COMMAND SharpIBChannel-6 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
  COMMAND SharpIBChannel-7 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
  COMMAND SharpIBChannel-8 --write-csv >> ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
  BYPRODUCTS ${CMAKE_SOURCE_DIR}/scaling/output/SharpIBChannel-Scaling.csv
)
