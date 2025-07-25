INC = -Isrc/

ifdef IGOR_DIR
  IGOR_INC = -I${IGOR_DIR}
else
  ${error "Need to define the path to Igor library in `IGOR_DIR`."}
endif

ifdef HYPRE_SERIAL_DIR
  HYPRE_INC = -I${HYPRE_SERIAL_DIR}/include
  HYPRE_LIB = -L${HYPRE_SERIAL_DIR}/lib -lHYPRE
else
  ${error "Need to define the path to HYPRE configured to not use MPI in `HYPRE_SERIAL_DIR`."}
endif

ifdef IRL_DIR
  IRL_INC = -I${IRL_DIR}/include
  IRL_LIB = -L${IRL_DIR}/lib -lirl
else
  ${error "Need to define the path to interface reconstruction library (IRL) in `IRL_DIR`."}
endif

ifdef EIGEN_DIR
  EIGEN_INC = -I${EIGEN_DIR}
else
  ${error "Need to define the path to Eigen linear algebra library in `EIGEN_DIR`."}
endif
