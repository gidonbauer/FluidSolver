#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_HPP_

#include <HYPRE_struct_ls.h>
#include <HYPRE_utilities.h>

#include <Igor/Math.hpp>
#include <Igor/MdArray.hpp>

#include "Config.hpp"
#include "FS.hpp"

struct PS {
  HYPRE_StructGrid grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix matrix;
  HYPRE_StructVector rhs;
  HYPRE_StructVector sol;
  HYPRE_StructSolver solver;
  HYPRE_StructSolver precond;
};

constexpr int COMM = -1;

[[nodiscard]] auto init_pressure_correction() -> PS {
  HYPRE_Initialize();

  PS ps{};

  // = Create structured grid ======================================================================
  {
    HYPRE_StructGridCreate(COMM, NDIMS, &ps.grid);
    std::array<HYPRE_Int, 2> ilower = {0, 0};
    std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
    HYPRE_StructGridSetExtents(ps.grid, ilower.data(), iupper.data());
    HYPRE_StructGridAssemble(ps.grid);
  }

  // = Create stencil ==============================================================================
  constexpr HYPRE_Int stencil_size                                       = 5;
  std::array<std::array<HYPRE_Int, NDIMS>, stencil_size> stencil_offsets = {
      std::array<HYPRE_Int, NDIMS>{0, 0},
      std::array<HYPRE_Int, NDIMS>{-1, 0},
      std::array<HYPRE_Int, NDIMS>{1, 0},
      std::array<HYPRE_Int, NDIMS>{0, -1},
      std::array<HYPRE_Int, NDIMS>{0, 1},
  };
  HYPRE_StructStencilCreate(NDIMS, stencil_size, &ps.stencil);
  for (HYPRE_Int i = 0; i < stencil_size; ++i) {
    HYPRE_StructStencilSetElement(
        ps.stencil, i, stencil_offsets[static_cast<size_t>(i)].data());  // NOLINT
  }

  // = Setup struct matrix =========================================================================
  // TODO: Assumes equidistant spacing in x- and y-direction respectively
  const auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  const auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

  HYPRE_StructMatrixCreate(COMM, ps.grid, ps.stencil, &ps.matrix);
  HYPRE_StructMatrixInitialize(ps.matrix);

  // Interior points
  {
    std::vector<double> matrix_values(static_cast<size_t>((NX - 2) * (NY - 2) * stencil_size));
    std::array<HYPRE_Int, stencil_size> stencil_indices = {0, 1, 2, 3, 4};
    for (size_t i = 0; i < matrix_values.size(); i += stencil_size) {
      matrix_values[i + 0] = -2.0 / Igor::sqr(dx) - 2.0 / Igor::sqr(dy);  // NOLINT
      matrix_values[i + 1] = 1.0 / Igor::sqr(dx);                         // NOLINT
      matrix_values[i + 2] = 1.0 / Igor::sqr(dx);                         // NOLINT
      matrix_values[i + 3] = 1.0 / Igor::sqr(dy);                         // NOLINT
      matrix_values[i + 4] = 1.0 / Igor::sqr(dy);                         // NOLINT
    }

    std::array<HYPRE_Int, 2> ilower = {1, 1};
    std::array<HYPRE_Int, 2> iupper = {NX - 2, NY - 2};
    HYPRE_StructMatrixSetBoxValues(ps.matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   stencil_size,
                                   stencil_indices.data(),
                                   matrix_values.data());
  }

  // Boundary points
  // Left:
  {
    std::array<HYPRE_Int, NDIMS> ilower = {0, 1};
    std::array<HYPRE_Int, NDIMS> iupper = {0, NY - 2};
    std::vector<HYPRE_Int> stencil_indices(stencil_size * (NY - 2));
    std::vector<double> values(stencil_indices.size());
    for (size_t i = 0; i < stencil_indices.size(); i += stencil_size) {
      stencil_indices[i + 0] = 0;
      stencil_indices[i + 1] = 1;
      stencil_indices[i + 2] = 2;
      stencil_indices[i + 3] = 3;
      stencil_indices[i + 4] = 4;

      values[i + 0] = -1.0 / Igor::sqr(dx) - 2.0 / Igor::sqr(dy);  // NOLINT
      values[i + 1] = 0.0;                                         // NOLINT
      values[i + 2] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 3] = 1.0 / Igor::sqr(dy);                         // NOLINT
      values[i + 4] = 1.0 / Igor::sqr(dy);                         // NOLINT
    }
    HYPRE_StructMatrixSetBoxValues(ps.matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   stencil_size,
                                   stencil_indices.data(),
                                   values.data());
  }
  // Right:
  {
    std::array<HYPRE_Int, NDIMS> ilower = {NX - 1, 1};
    std::array<HYPRE_Int, NDIMS> iupper = {NX - 1, NY - 2};
    std::vector<HYPRE_Int> stencil_indices(stencil_size * (NY - 2));
    std::vector<double> values(stencil_indices.size());
    for (size_t i = 0; i < stencil_indices.size(); i += stencil_size) {
      stencil_indices[i + 0] = 0;
      stencil_indices[i + 1] = 1;
      stencil_indices[i + 2] = 2;
      stencil_indices[i + 3] = 3;
      stencil_indices[i + 4] = 4;

      values[i + 0] = -1.0 / Igor::sqr(dx) - 2.0 / Igor::sqr(dy);  // NOLINT
      values[i + 1] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 2] = 0.0;                                         // NOLINT
      values[i + 3] = 1.0 / Igor::sqr(dy);                         // NOLINT
      values[i + 4] = 1.0 / Igor::sqr(dy);                         // NOLINT
    }
    HYPRE_StructMatrixSetBoxValues(ps.matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   stencil_size,
                                   stencil_indices.data(),
                                   values.data());
  }
  // Bottom:
  {
    std::array<HYPRE_Int, NDIMS> ilower = {1, 0};
    std::array<HYPRE_Int, NDIMS> iupper = {NX - 2, 0};
    std::vector<HYPRE_Int> stencil_indices(stencil_size * (NX - 2));
    std::vector<double> values(stencil_indices.size());
    for (size_t i = 0; i < stencil_indices.size(); i += stencil_size) {
      stencil_indices[i + 0] = 0;
      stencil_indices[i + 1] = 1;
      stencil_indices[i + 2] = 2;
      stencil_indices[i + 3] = 3;
      stencil_indices[i + 4] = 4;

      values[i + 0] = -2.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy);  // NOLINT
      values[i + 1] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 2] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 3] = 0.0;                                         // NOLINT
      values[i + 4] = 1.0 / Igor::sqr(dy);                         // NOLINT
    }
    HYPRE_StructMatrixSetBoxValues(ps.matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   stencil_size,
                                   stencil_indices.data(),
                                   values.data());
  }
  // Top:
  {
    std::array<HYPRE_Int, NDIMS> ilower = {1, NY - 1};
    std::array<HYPRE_Int, NDIMS> iupper = {NX - 2, NY - 1};
    std::vector<HYPRE_Int> stencil_indices(stencil_size * (NX - 2));
    std::vector<double> values(stencil_indices.size());
    for (size_t i = 0; i < stencil_indices.size(); i += stencil_size) {
      stencil_indices[i + 0] = 0;
      stencil_indices[i + 1] = 1;
      stencil_indices[i + 2] = 2;
      stencil_indices[i + 3] = 3;
      stencil_indices[i + 4] = 4;

      values[i + 0] = -2.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy);  // NOLINT
      values[i + 1] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 2] = 1.0 / Igor::sqr(dx);                         // NOLINT
      values[i + 3] = 1.0 / Igor::sqr(dy);                         // NOLINT
      values[i + 4] = 0.0;                                         // NOLINT
    }
    HYPRE_StructMatrixSetBoxValues(ps.matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   stencil_size,
                                   stencil_indices.data(),
                                   values.data());
  }

  // Bottom-left
  {
    std::array<HYPRE_Int, NDIMS> idx       = {0, 0};
    std::vector<HYPRE_Int> stencil_indices = {0, 1, 2, 3, 4};
    std::vector<double> values             = {
        -1.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy),
        0.0,
        1.0 / Igor::sqr(dx),
        0.0,
        1.0 / Igor::sqr(dy),
    };
    HYPRE_StructMatrixSetValues(
        ps.matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
  }
  // Bottom-right
  {
    std::array<HYPRE_Int, NDIMS> idx       = {NX - 1, 0};
    std::vector<HYPRE_Int> stencil_indices = {0, 1, 2, 3, 4};
    std::vector<double> values             = {
        -1.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy),
        1.0 / Igor::sqr(dx),
        0.0,
        0.0,
        1.0 / Igor::sqr(dy),
    };
    HYPRE_StructMatrixSetValues(
        ps.matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
  }
  // Top-left
  {
    std::array<HYPRE_Int, NDIMS> idx       = {0, NY - 1};
    std::vector<HYPRE_Int> stencil_indices = {0, 1, 2, 3, 4};
    std::vector<double> values             = {
        -1.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy),
        0.0,
        1.0 / Igor::sqr(dx),
        1.0 / Igor::sqr(dy),
        0.0,
    };
    HYPRE_StructMatrixSetValues(
        ps.matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
  }
  // Top-right
  {
    std::array<HYPRE_Int, NDIMS> idx       = {NX - 1, NY - 1};
    std::vector<HYPRE_Int> stencil_indices = {0, 1, 2, 3, 4};
    std::vector<double> values             = {
        -1.0 / Igor::sqr(dx) - 1.0 / Igor::sqr(dy),
        1.0 / Igor::sqr(dx),
        0.0,
        1.0 / Igor::sqr(dy),
        0.0,
    };
    HYPRE_StructMatrixSetValues(
        ps.matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
  }

  HYPRE_StructMatrixAssemble(ps.matrix);

  // = Create right-hand side ======================================================================
  HYPRE_StructVectorCreate(COMM, ps.grid, &ps.rhs);
  HYPRE_StructVectorInitialize(ps.rhs);

  // = Create solution vector ======================================================================
  HYPRE_StructVectorCreate(COMM, ps.grid, &ps.sol);
  HYPRE_StructVectorInitialize(ps.sol);

  // = Create solver ===============================================================================
#ifdef FS_USE_PCG_SOLVER
  HYPRE_StructPCGCreate(COMM, &ps.solver);
  HYPRE_StructPCGSetTol(ps.solver, 1e-8);
#ifdef FS_HYPRE_VERBOSE
  HYPRE_StructPCGSetPrintLevel(ps.solver, 2);
  HYPRE_StructPCGSetLogging(ps.solver, 1);
#endif  // FS_HYPRE_VERBOSE
#else
  HYPRE_StructFlexGMRESCreate(COMM, &ps.solver);
  HYPRE_StructFlexGMRESSetTol(ps.solver, 1e-8);
  HYPRE_StructFlexGMRESSetMaxIter(ps.solver, 500);
#ifdef FS_HYPRE_VERBOSE
  HYPRE_StructFlexGMRESSetPrintLevel(ps.solver, 2);
  HYPRE_StructFlexGMRESSetLogging(ps.solver, 1);
#endif  // FS_HYPRE_VERBOSE
#endif

  // = Create preconditioner =======================================================================
#ifdef FS_USE_PFMG_PRECOND

  HYPRE_StructPFMGCreate(COMM, &ps.precond);  // NOLINT
  HYPRE_StructPFMGSetMaxIter(ps.precond, 1);
  HYPRE_StructPFMGSetTol(ps.precond, 0.0);
  HYPRE_StructPFMGSetZeroGuess(ps.precond);
  HYPRE_StructPFMGSetRAPType(ps.precond, 0);
  HYPRE_StructPFMGSetRelaxType(ps.precond, 1);
  HYPRE_StructPFMGSetNumPreRelax(ps.precond, 1);
  HYPRE_StructPFMGSetNumPostRelax(ps.precond, 1);
  HYPRE_StructPFMGSetSkipRelax(ps.precond, 0);
#ifdef FS_USE_PCG_SOLVER
  HYPRE_StructPCGSetPrecond(ps.solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, ps.precond);
#else
  HYPRE_StructFlexGMRESSetPrecond(
      ps.solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, ps.precond);
#endif

#else

  HYPRE_StructSMGCreate(COMM, &ps.precond);  // NOLINT
  HYPRE_StructSMGSetMaxIter(ps.precond, 1);
  HYPRE_StructSMGSetTol(ps.precond, 0.0);
  HYPRE_StructSMGSetZeroGuess(ps.precond);
  HYPRE_StructSMGSetNumPreRelax(ps.precond, 1);
  HYPRE_StructSMGSetNumPostRelax(ps.precond, 1);
  HYPRE_StructSMGSetMemoryUse(ps.precond, 0);
#ifdef FS_USE_PCG_SOLVER
  HYPRE_StructPCGSetPrecond(ps.solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, ps.precond);
#else
  HYPRE_StructFlexGMRESSetPrecond(
      ps.solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, ps.precond);
#endif

#endif

  const HYPRE_Int error_flag = HYPRE_GetError();
  if (error_flag != 0) { Igor::Panic("An error occured in HYPRE."); }

  return ps;
}

// -------------------------------------------------------------------------------------------------
void finalize_pressure_correction(PS& ps) {
#ifdef FS_USE_PFMG_PRECOND
  HYPRE_StructPFMGDestroy(ps.precond);
#else
  HYPRE_StructSMGDestroy(ps.precond);
#endif

#ifdef FS_USE_PCG_SOLVER
  HYPRE_StructPCGDestroy(ps.solver);
#else
  HYPRE_StructFlexGMRESDestroy(ps.solver);
#endif

  HYPRE_StructVectorDestroy(ps.sol);
  HYPRE_StructVectorDestroy(ps.rhs);
  HYPRE_StructMatrixDestroy(ps.matrix);
  HYPRE_StructStencilDestroy(ps.stencil);
  HYPRE_StructGridDestroy(ps.grid);
  HYPRE_Finalize();
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto calc_pressure_correction(const FS& fs,
                                            const Igor::MdArray<Float, CENTERED_EXTENT>& div,
                                            const PS& ps,
                                            Float dt,
                                            Igor::MdArray<Float, CENTERED_EXTENT>& resP) -> bool {
  static std::array<char, 1024UZ> buffer{};
  static_cast<void>(fs);
  HYPRE_Int ierr = 0;
  bool res       = true;

  static Igor::MdArray<Float, CENTERED_EXTENT, std::layout_left> rhs(resP.extent(0),
                                                                     resP.extent(1));
  [[maybe_unused]] Float mean_rhs = 0.0;
  for (size_t i = 0; i < resP.extent(0); ++i) {
    for (size_t j = 0; j < resP.extent(1); ++j) {
      rhs[i, j] = fs.rho[i, j] * div[i, j] / dt;
      mean_rhs += rhs[i, j];
    }
  }
  mean_rhs /= rhs.size();
#ifndef FS_PRESSURE_CORR_NO_SHIFT_RHS
  for (size_t i = 0; i < resP.extent(0); ++i) {
    for (size_t j = 0; j < resP.extent(1); ++j) {
      rhs[i, j] -= mean_rhs;
    }
  }
#endif  // FS_PRESSURE_CORR_NO_SHIFT_RHS

  std::array<HYPRE_Int, 2> ilower = {0, 0};
  std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
  HYPRE_StructVectorSetBoxValues(ps.rhs, ilower.data(), iupper.data(), rhs.get_data());
  HYPRE_StructVectorAssemble(ps.rhs);

  Float final_residual = -1.0;
#ifdef FS_USE_PCG_SOLVER
  HYPRE_StructPCGSetup(ps.solver, ps.matrix, ps.rhs, ps.sol);
  ierr = HYPRE_StructPCGSolve(ps.solver, ps.matrix, ps.rhs, ps.sol);
  HYPRE_StructPCGGetFinalRelativeResidualNorm(ps.solver, &final_residual);
#else
  HYPRE_StructFlexGMRESSetup(ps.solver, ps.matrix, ps.rhs, ps.sol);
  ierr = HYPRE_StructFlexGMRESSolve(ps.solver, ps.matrix, ps.rhs, ps.sol);
  HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(ps.solver, &final_residual);
#endif

  IGOR_DEBUG_PRINT(final_residual);
  if (final_residual > 1e-6) {
    Igor::Warn("Residual pressure correction = {}", final_residual);
    res = false;
  }

  for (size_t i = 0; i < resP.extent(0); ++i) {
    for (size_t j = 0; j < resP.extent(1); ++j) {
      std::array<HYPRE_Int, NDIMS> idx = {static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
      ierr = HYPRE_StructVectorGetValues(ps.sol, idx.data(), &resP[i, j]);
    }
  }

  if (ierr != 0) {
    HYPRE_DescribeError(ierr, buffer.data());
    Igor::Warn("Could not solve the system successfully: {}", buffer.data());
    HYPRE_ClearError(ierr);
    res = false;
  }

  return res;
}

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
