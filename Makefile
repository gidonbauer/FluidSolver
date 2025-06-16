SRC = src/FluidSolver.cpp src/Config.hpp src/FS.hpp src/IO.hpp src/Operators.hpp src/PressureCorrection.hpp src/Container.hpp

CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
CXX_RELEASE_FLAGS = -march=native -ffast-math -O3 -DIGOR_NDEBUG
CXX_DEBUG_FLAGS = -O0 -g -D_GLIBCXX_DEBUG
CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

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

release: CXX_FLAGS += ${CXX_RELEASE_FLAGS}
release: FluidSolver

debug: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
debug: FluidSolver

sanitize: CXX_FLAGS += ${CXX_DEBUG_FLAGS} ${CXX_SANITIZER_FLAGS}
sanitize: FluidSolver

profile: CXX_FLAGS += ${CXX_RELEASE_FLAGS} -g -pg
profile: PROFILE_LIB = -L/opt/homebrew/opt/gperftools/lib -lprofiler
profile: FluidSolver

FluidSolver: ${SRC}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${PROFILE_LIB}

clean:
	${RM} -r FluidSolver FluidSolver.dSYM

.PHONY: release debug clean
