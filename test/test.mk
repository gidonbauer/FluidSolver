TESTS = LaminarChannel Couette TaylorGreenVortex \
				ConstantVelocityVOF LinearVelocityVOF TaylorGreenVortexVOF \
				Operators

test: ${addprefix test/, ${TESTS}} ${addprefix test-, ${TESTS}}

test-%: test/%
	@printf "\033[32m[TEST]\033[0m Running test case $*...\n"
	@$< && printf "\033[32m[PASS]\033[0m $* finished successfully.\n\n" || printf "\033[31m[FAIL]\033[0m $* failed.\n\n"

test/%: test/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB}

clean-test:
	${RM} -r ${addprefix test/, ${TESTS}} ${addsuffix .dSYM, ${addprefix test/, ${TESTS}}}

.PHONY: test clean-test
