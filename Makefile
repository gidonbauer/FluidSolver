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

TARGETS = IncompSolver VOF Curvature TwoPhaseSolver IB

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

%: examples/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

TwoPhaseSolver: examples/TwoPhaseSolver.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

clean: clean-test
	${RM} -r ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

include test/test.mk

.PHONY: all clean
