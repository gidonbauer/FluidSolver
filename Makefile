HEADERS = src/Container.hpp          \
          src/FS.hpp                 \
          src/IO.hpp                 \
          src/Operators.hpp          \
          src/PressureCorrection.hpp \
          src/IR.hpp                 \
          src/VOF.hpp                \
          src/Monitor.hpp            \
          src/Quadrature.hpp         \
          src/QuadratureTables.hpp   \
          src/VTKWriter.hpp          \
          src/Curvature.hpp

TARGETS = IncompSolver VOF Curvature TwoPhaseSolver

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

IncompSolver: examples/IncompSolver.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

VOF: examples/VOF.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

Curvature: examples/Curvature.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

# TwoPhaseSolver: examples/TwoPhaseSolver.cpp ${HEADERS}
# 	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

TwoPhaseSolver: examples/TwoPhaseSolver.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} \
		${INC} \
		${IGOR_INC} \
		-I${HOME}/opt/hypre-2.33.0-Cuda/src/hypre/include \
		${IRL_INC} \
		${EIGEN_INC} \
		-I${HOME}/opt/fmt-11.2.0/fmt/include -DIGOR_USE_FMT \
		-o $@ $< \
		-L${HOME}/opt/hypre-2.33.0-Cuda/src/hypre/lib -lHYPRE \
		${IRL_LIB} \
		-L${HOME}/opt/fmt-11.2.0/fmt/lib64 -lfmt \
		-lcudart -lcurand -lcublas -lcusolver -lcusparse \
		-Wl,-rpath /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/intel-compilers/2024.2.0/compiler/latest/lib \
		-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/intel-compilers/2024.2.0/compiler/latest/lib -lirc -limf

clean: clean-test
	${RM} -r ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

include test/test.mk

.PHONY: all clean
