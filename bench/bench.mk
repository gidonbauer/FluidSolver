BENCHMARKS = dmomdt

bench: ${addprefix bench/, ${BENCHMARKS}} ${addprefix bench-, ${BENCHMARKS}}

bench-%: bench/%
	@printf "\033[32m[TEST]\033[0m Running benchmark $*...\n"
	@$<

bench/%: bench/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

clean-bench:
	${RM} -r ${addprefix bench/, ${BENCHMARKS}} ${addsuffix .dSYM, ${addprefix bench/, ${BENCHMARKS}}}

.PHONY: test clean-test
