INC = -Isrc/

ifdef IGOR_DIR
  IGOR_INC = -I${IGOR_DIR}
else
  ${error "Need to define the path to Igor library in `IGOR_DIR`."}
endif

ifdef HYPRE_OPENMP_DIR
  HYPRE_INC = -I${HYPRE_OPENMP_DIR}/include
  HYPRE_LIB = -L${HYPRE_OPENMP_DIR}/lib -lHYPRE
else
  ${error "Need to define the path to HYPRE configured to not use MPI in `HYPRE_OPENMP_DIR`."}
endif

ifdef IRL_DIR
  IRL_INC = -isystem${IRL_DIR}/include
  IRL_LIB = -L${IRL_DIR}/lib -lirl
else
  ${error "Need to define the path to interface reconstruction library (IRL) in `IRL_DIR`."}
endif

ifdef EIGEN_DIR
  EIGEN_INC = -I${EIGEN_DIR}
else
  ${error "Need to define the path to Eigen linear algebra library in `EIGEN_DIR`."}
endif

ifdef HDF_DIR
  HDF_INC = -I${HDF_DIR}/include
  HDF_LIB = -L${HDF_DIR}/lib -lhdf5_hl_cpp -lhdf5_hl -lhdf5
else
  HDF_INC = -DFS_DISABLE_HDF
endif

ifdef $(HOSTNAME)
  HOST_NAME := $(strip $(HOSTNAME))
else ifdef $(HOST)
  HOST_NAME := $(strip $(HOST))
else
  HOST_NAME := $(shell hostname -f)
endif

ifeq ($(findstring hpc.itc.rwth-aachen.de, ${HOST_NAME}), hpc.itc.rwth-aachen.de)
  INTEL_RT_DIR = /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/intel-compilers/2024.2.0/compiler/latest
  INTEL_RT_LIB = -Wl,-rpath ${INTEL_RT_DIR}/lib -L${INTEL_RT_DIR}/lib -lirc -limf
endif

ifeq (${BASENAME_CXX}, nvc++)
  CUDA_LIB = -lcudart -lcurand -lcublas -lcusolver -lcusparse
endif
