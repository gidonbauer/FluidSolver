TESTS = LaminarChannel Couette TaylorGreenVortex

test/%: test/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} -o $@ $< ${HYPRE_LIB}

run-%: test/%
	@echo "Running test case $*..."
	@$< && echo "$* finished successfully." || echo "$* failed."
	@echo ""

test: CXX_FLAGS += ${CXX_DEBUG_FLAGS}
test: ${addprefix test/, ${TESTS}} ${addprefix run-, ${TESTS}}

clean-test:
	${RM} -r ${addprefix test/, ${TESTS}} ${addsuffix .dSYM, ${addprefix test/, ${TESTS}}}

.PHONY: test clean-test
