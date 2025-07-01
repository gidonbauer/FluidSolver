TESTS = LaminarChannel Couette TaylorGreenVortex ConstantVelocityVOF LinearVelocityVOF EvalFlowFieldAt

test: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
test: ${addprefix test/, ${TESTS}} ${addprefix run-, ${TESTS}}

test-fast: CXX_FLAGS += ${CXX_RELEASE_FLAGS}
test-fast: ${addprefix test/, ${TESTS}} ${addprefix run-, ${TESTS}}

run-%: test/%
	@echo "Running test case $*..."
	@$< && printf "\033[32m[OK]\033[0m $* finished successfully." || printf "\033[31m[FAIL]\033[0m $* failed."
	@echo ""

test/%: test/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} -o $@ $< ${HYPRE_LIB}

test/ConstantVelocityVOF: test/ConstantVelocityVOF.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

test/LinearVelocityVOF: test/LinearVelocityVOF.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

clean-test:
	${RM} -r ${addprefix test/, ${TESTS}} ${addsuffix .dSYM, ${addprefix test/, ${TESTS}}}

.PHONY: test clean-test
