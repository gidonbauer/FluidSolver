BENCHMARKS = dmomdt-no-fuse   \
             dmomdt-fuse-flux \
             dmomdt-fuse-all  \
             update-velo-fuse

bench: ${addprefix bench/, ${BENCHMARKS}} ${addprefix bench-, ${BENCHMARKS}}

bench-%: bench/%
	@printf "\033[94m[BENCH]\033[0m Running benchmark $*...\n"
	@$<
	@echo ""

bench/dmomdt-no-fuse: bench/dmomdt.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

bench/dmomdt-fuse-flux: bench/dmomdt.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} -DFS_FUSE_MOM_FLUX ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

bench/dmomdt-fuse-all: bench/dmomdt.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} -DFS_FUSE_MOM_ALL ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

bench/%: bench/%.cpp ${HEADERS}
	${CXX} ${CXX_FLAGS} ${CXX_OPENMP_FLAGS} ${INC} ${IGOR_INC} ${HYPRE_INC} ${IRL_INC} ${EIGEN_INC} ${HDF_INC} -o $@ $< ${HYPRE_LIB} ${IRL_LIB} ${HDF_LIB} ${INTEL_RT_LIB} ${CUDA_LIB}

clean-bench:
	${RM} -r ${addprefix bench/, ${BENCHMARKS}} ${addsuffix .dSYM, ${addprefix bench/, ${BENCHMARKS}}}

.PHONY: test clean-test
