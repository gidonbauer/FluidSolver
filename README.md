# Fluid Solver

A rudimentary solver for the incompressible Navier Stokes Equation based on a finite differences discretization on a staggered mesh.

## Quickstart

Build the executable with `make` and run it:
```console
$ make
$ ./FluidSolver
```
The output is written into `output` in VTK-format and can be inspected with Paraview.

The problem that is solver can be modified in the `Config.hpp` file in the `src` directory.

## Dependencies

The dependencies are managed manually via the enviroment variables `IGOR_DIR`, `HYPRE_SERIAL_DIR`, `IRL_DIR`, and `EIGEN_DIR`.

- [Igor](https://github.com/gidonbauer/Igor)
- [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods): Needs to be configured to **not** use MPI
- [Interface Reconstruction Library](https://github.com/robert-chiodi/interface-reconstruction-library)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
