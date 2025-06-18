TESTS = LaminarChannel Couette TaylorGreenVortex

test: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
test: ${addprefix test/, ${TESTS}} ${addprefix run-, ${TESTS}}

run-%: test/%
	@echo "Running test case $*..."
	@$< && echo "$* finished successfully." || echo "$* failed."
	@echo ""

test/%: test/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} -o $@ $< ${HYPRE_LIB}

clean-test:
	${RM} -r ${addprefix test/, ${TESTS}} ${addsuffix .dSYM, ${addprefix test/, ${TESTS}}}

.PHONY: test clean-test
