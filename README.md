# Fluid Solver

A rudimentary solver for the incompressible Navier Stokes Equation based on a finite differences discretization on a staggered mesh.

## Quickstart

Build the executable with `make` and run it:
```console
$ make
$ ./FluidSolver
```
The output is written into `output` in VTK-format and can be instpected with Paraview.

## Dependencies

The dependencies are managed manually via the enviroment variables `IGOR_DIR` and `HYPRE_OMP_DIR`.

- [Igor](https://github.com/gidonbauer/Igor)
- [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods): Needs to be configured to use OpenMP and **not** MPI
- [OpenMP](https://www.openmp.org/): Should come with the C++ compiler
