# Fluid Solver

A solver for the two-dimensional, incompressible, two-phase Navier-Stokes Equations based on a finite differences discretization on a staggered mesh using a Volume-of-Fluid method.

## Quickstart

Configure and build using `cmake`:
```console
$ cmake -Bbuild
$ cd build
$ make [-j]
```
Examples cases are in the `example` folder, tests are located in `test` and some benchmarks in `bench`.
The examples can be run from within the build folder, e.g.
```console
$ ./example/TwoPhaseSolver
```
The output is written into the folder `output/<case name>` relative to the current working directory.
It can be written in VTK or XDMF2-format and inspected with ParaView. The latter one uses HDF5 as a backend to store the date.
> [!WARNING]  
> The data in the HDF5 files is stored in Fortran-order even though HDF5 expects C-order.
> This is done due to constraints with ParaView and must be kept in mind when reading the data with other programs.

## Tests

Test cases are implemented in the `test` folder and can be run by the command `ctest` in the build folder.
The output of the test cases are written into `test/output/<test case>` and can be inspected with ParaView.

The implemented test cases are
- Laminar channel flow
- Couette flow
- Taylor-Green vortex
- VOF advection with constant velocity field
- VOF advection with linear velocity field
- VOF advection with Taylor-Green vortex
- Moving drop with high density difference
- Stationary drop with surface tension forces
- Additional test on the source code

## Dependencies

The dependencies are managed via git-submodules.
The Interface Reconstruction Library has further dependencies on Eigen and Abseil. Eigen might need to be installed separately.
HDF5 is optional and must also be installed separately.

- [OpenMP](https://www.openmp.org/)
- [Igor](https://github.com/gidonbauer/Igor)
- [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)
- [Interface Reconstruction Library](https://github.com/robert-chiodi/interface-reconstruction-library)
- [HDF5](https://github.com/HDFGroup/hdf5)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
