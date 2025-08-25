HEADERS = src/BoundaryConditions.hpp \
          src/Container.hpp          \
          src/Curvature.hpp          \
          src/ForEach.hpp            \
          src/FS.hpp                 \
          src/IO.hpp                 \
          src/IotaIter.hpp           \
          src/IR.hpp                 \
          src/Macros.hpp             \
          src/Monitor.hpp            \
          src/Operators.hpp          \
          src/PressureCorrection.hpp \
          src/Quadrature.hpp         \
          src/QuadratureTables.hpp   \
          src/StdparOpenMP.hpp       \
          src/Utility.hpp            \
          src/VOF.hpp                \
          src/VTKWriter.hpp          \
          src/XDMFWriter.hpp


TARGETS = IncompSolver VOF Curvature TwoPhaseSolver IB PhaseChange

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

INTEL_RT_DIR = /cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/intel-compilers/2024.2.0/compiler/latest
INTEL_RT_LIBS = -Wl,-rpath ${INTEL_RT_DIR}/lib -L${INTEL_RT_DIR}/lib -lirc -limf

CUDA_LIBS = -lcudart -lcurand -lcublas -lcusolver -lcusparse

%: examples/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIBS} ${CUDA_LIBS}

clean: clean-test
	${RM} -r ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

include test/test.mk

.PHONY: all clean
