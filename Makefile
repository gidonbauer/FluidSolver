HEADERS = src/Container.hpp src/FS.hpp src/IO.hpp src/Operators.hpp src/PressureCorrection.hpp
TARGETS = IncompSolver VOF

BASENAME_CXX = ${notdir ${CXX}}
ifeq (${BASENAME_CXX}, clang++)
	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -march=native -ffast-math -O3 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g -D_GLIBCXX_DEBUG
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
else ifeq (${BASENAME_CXX}, ${filter ${BASENAME_CXX}, g++ g++-15})
	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -march=native -ffast-math -O3 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g -D_GLIBCXX_DEBUG
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
else ifeq (${BASENAME_CXX}, icpx)
	CXX_FLAGS = -Wall -Wextra -pedantic -Wshadow -Wconversion -std=c++23
	CXX_RELEASE_FLAGS = -O3 -ffast-math -xSSE4.2 -axCORE-AVX2,AVX -fp-model fast=2 -DIGOR_NDEBUG
	CXX_DEBUG_FLAGS = -O0 -g -D_GLIBCXX_DEBUG
	CXX_SANITIZER_FLAGS = -fsanitize=address,undefined
else
  ${error "Unknown C++ compiler `${CXX}`"}
endif

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
release: ${TARGETS}

debug: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
debug: ${TARGETS}

sanitize: CXX_FLAGS += ${CXX_DEBUG_FLAGS} ${CXX_SANITIZER_FLAGS}
sanitize: ${TARGETS}

score-p: CXX_FLAGS += ${CXX_RELEASE_FLAGS} -g
score-p: CXX := scorep ${CXX}
score-p: ${TARGETS}

IncompSolver: src/IncompSolver.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} -o $@ $< ${HYPRE_LIB}

VOF: src/VOF.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

clean:
	${RM} -r ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

.PHONY: release debug clean
