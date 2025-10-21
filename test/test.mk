TESTS = LaminarChannel_0     \
        LaminarChannel_1     \
        Couette              \
        TaylorGreenVortex    \
        ConstantVelocityVOF  \
        LinearVelocityVOF    \
        TaylorGreenVortexVOF \
        MovingDrop           \
        StationaryDrop       \
        Operators            \
        Container            \
        Utility              \
        OffloadGPU

test: ${addprefix test/, ${TESTS}} ${addprefix test-, ${TESTS}}

test-%: test/%
	@printf "\033[32m[TEST]\033[0m Running test case $*...\n"
	@OMP_NUM_THREADS=4 $< && printf "\033[32m[PASS]\033[0m $* finished successfully.\n\n" || printf "\033[31m[FAIL]\033[0m $* failed.\n\n"

test/%: test/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

test/LaminarChannel_0: test/LaminarChannel.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} -DLC_U_INIT=0 ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

test/LaminarChannel_1: test/LaminarChannel.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} -DLC_U_INIT=1 ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

clean-test:
	${RM} -r ${addprefix test/, ${TESTS}} ${addsuffix .dSYM, ${addprefix test/, ${TESTS}}}

.PHONY: test clean-test
