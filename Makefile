SRC = src/FluidSolver.cpp src/Config.hpp src/FS.hpp src/IO.hpp src/Operators.hpp src/PressureCorrection.hpp

CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
CXX_RELEASE_FLAGS = -march=native -ffast-math -O3
CXX_DEBUG_FLAGS = -O0 -g
CXX_SANITIZER_FLAGS = -fsanitize=address,undefined

INC = -Isrc/

ifdef IGOR_DIR
  IGOR_INC = -I${IGOR_DIR}
else
  ${error "Need to define the path to Igor library in `IGOR_DIR`."}
endif

ifdef HYPRE_OMP_DIR
  HYPRE_INC = -I${HYPRE_OMP_DIR}/include
  HYPRE_LIB = -L${HYPRE_OMP_DIR}/lib -lHYPRE
else
  ${error "Need to define the path to HYPRE configured to use OpenMP instead of MPI in `HYPRE_OMP_DIR`."}
endif

OMP_INC = -fopenmp
# OMP_INC = -I/opt/homebrew/opt/libomp/include -fopenmp
# OMP_LIB = -L/opt/homebrew/opt/libomp/lib -lomp

release: CXX_FLAGS += ${CXX_RELEASE_FLAGS}
release: FluidSolver

debug: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
debug: FluidSolver

sanitize: CXX_FLAGS += ${CXX_DEBUG_FLAGS} ${CXX_SANITIZER_FLAGS}
sanitize: FluidSolver

FluidSolver: ${SRC}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${OMP_INC} -o $@ $< ${HYPRE_LIB} ${OMP_LIB}

clean:
	${RM} -r FluidSolver FluidSolver.dSYM

clean-output:
	${RM} -r output

.PHONY: release debug clean clean-output
