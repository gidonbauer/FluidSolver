HEADERS = src/BoundaryConditions.hpp \
          src/Container.hpp          \
          src/Curvature.hpp          \
          src/ForEach.hpp            \
          src/FS.hpp                 \
          src/IO.hpp                 \
          src/IR.hpp                 \
          src/Macros.hpp             \
          src/Monitor.hpp            \
          src/Operators.hpp          \
          src/PressureCorrection.hpp \
          src/Quadrature.hpp         \
          src/QuadratureTables.hpp   \
          src/Utility.hpp            \
          src/VOF.hpp                \
          src/VTKWriter.hpp          \
          src/XDMFWriter.hpp


TARGETS = IncompSolver VOF Curvature TwoPhaseSolver IB PhaseChange

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

POOLSTL_DIR = ${HOME}/opt/poolSTL
POOLSTL_INC = -I${POOLSTL_DIR}/include

%: examples/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} ${POOLSTL_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB}

clean: clean-test
	${RM} -r ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

include test/test.mk

.PHONY: all clean
