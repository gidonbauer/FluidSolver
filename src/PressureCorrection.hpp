#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_HPP_

#include <HYPRE_struct_ls.h>
#include <HYPRE_utilities.h>

#include <Igor/Math.hpp>

#include "Config.hpp"
#include "FS.hpp"

class PS {
  constexpr static int COMM = -1;

  HYPRE_StructGrid grid{};
  HYPRE_StructStencil stencil{};
  HYPRE_StructMatrix matrix{};
  HYPRE_StructVector rhs{};
  HYPRE_StructVector sol{};
  HYPRE_StructSolver solver{};
  HYPRE_StructSolver precond{};

 public:
  constexpr PS() noexcept {
    HYPRE_Initialize();

    // = Create structured grid ====================================================================
    {
      HYPRE_StructGridCreate(COMM, NDIMS, &grid);
      std::array<HYPRE_Int, 2> ilower = {0, 0};
      std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
      HYPRE_StructGridSetExtents(grid, ilower.data(), iupper.data());
      HYPRE_StructGridAssemble(grid);
    }

    // = Create stencil ============================================================================
    constexpr HYPRE_Int stencil_size                                       = 5;
    std::array<std::array<HYPRE_Int, NDIMS>, stencil_size> stencil_offsets = {
        std::array<HYPRE_Int, NDIMS>{0, 0},
        std::array<HYPRE_Int, NDIMS>{-1, 0},
        std::array<HYPRE_Int, NDIMS>{1, 0},
        std::array<HYPRE_Int, NDIMS>{0, -1},
        std::array<HYPRE_Int, NDIMS>{0, 1},
    };
    HYPRE_StructStencilCreate(NDIMS, stencil_size, &stencil);
    for (HYPRE_Int i = 0; i < stencil_size; ++i) {
      HYPRE_StructStencilSetElement(
          stencil, i, stencil_offsets[static_cast<size_t>(i)].data());  // NOLINT
    }

    // = Setup struct matrix =======================================================================
    // TODO: Assumes equidistant spacing in x- and y-direction respectively
    const auto dx  = (X_MAX - X_MIN) / static_cast<Float>(NX);
    const auto dy  = (Y_MAX - Y_MIN) / static_cast<Float>(NY);
    const auto vol = dx * dy;

    HYPRE_StructMatrixCreate(COMM, grid, stencil, &matrix);
    HYPRE_StructMatrixInitialize(matrix);

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
      for (auto& value : matrix_values) {
        value *= -vol;
      }

      std::array<HYPRE_Int, 2> ilower = {1, 1};
      std::array<HYPRE_Int, 2> iupper = {NX - 2, NY - 2};
      HYPRE_StructMatrixSetBoxValues(matrix,
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetBoxValues(matrix,
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetBoxValues(matrix,
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetBoxValues(matrix,
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetBoxValues(matrix,
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetValues(
          matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetValues(
          matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetValues(
          matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
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
      for (auto& value : values) {
        value *= -vol;
      }
      HYPRE_StructMatrixSetValues(
          matrix, idx.data(), stencil_size, stencil_indices.data(), values.data());
    }

    HYPRE_StructMatrixAssemble(matrix);

    // = Create right-hand side ====================================================================
    HYPRE_StructVectorCreate(COMM, grid, &rhs);
    HYPRE_StructVectorInitialize(rhs);

    // = Create solution vector ====================================================================
    HYPRE_StructVectorCreate(COMM, grid, &sol);
    HYPRE_StructVectorInitialize(sol);

    // = Create solver =============================================================================
    HYPRE_StructGMRESCreate(COMM, &solver);
    HYPRE_StructGMRESSetTol(solver, PRESSURE_TOL);
    HYPRE_StructGMRESSetMaxIter(solver, PRESSURE_MAX_ITER);
#ifdef FS_HYPRE_VERBOSE
    HYPRE_StructGMRESSetPrintLevel(solver, 2);
    HYPRE_StructGMRESSetLogging(solver, 1);
#endif  // FS_HYPRE_VERBOSE

    // = Create preconditioner =====================================================================
    HYPRE_StructSMGCreate(COMM, &precond);
    HYPRE_StructSMGSetMaxIter(precond, 1);
    HYPRE_StructSMGSetTol(precond, 0.0);
    HYPRE_StructSMGSetZeroGuess(precond);
    HYPRE_StructSMGSetNumPreRelax(precond, 1);
    HYPRE_StructSMGSetNumPostRelax(precond, 1);
    HYPRE_StructSMGSetMemoryUse(precond, 0);
    HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
    HYPRE_StructGMRESSetup(solver, matrix, rhs, sol);

    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) { Igor::Panic("An error occured in HYPRE."); }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr PS(const PS& other) noexcept                    = delete;
  constexpr PS(PS&& other) noexcept                         = delete;
  constexpr auto operator=(const PS& other) noexcept -> PS& = delete;
  constexpr auto operator=(PS&& other) noexcept -> PS&      = delete;

  // -----------------------------------------------------------------------------------------------
  constexpr ~PS() noexcept {
    HYPRE_StructSMGDestroy(precond);
    HYPRE_StructGMRESDestroy(solver);
    HYPRE_StructVectorDestroy(sol);
    HYPRE_StructVectorDestroy(rhs);
    HYPRE_StructMatrixDestroy(matrix);
    HYPRE_StructStencilDestroy(stencil);
    HYPRE_StructGridDestroy(grid);
    HYPRE_Finalize();
  }

  // -------------------------------------------------------------------------------------------------
  [[nodiscard]] auto
  solve(const FS& fs, const Matrix<Float, NX, NY>& div, Float dt, Matrix<Float, NX, NY>& resP)
      -> bool {
    static std::array<char, 1024UZ> buffer{};
    HYPRE_Int ierr = 0;
    bool res       = true;

    // TODO: Assumes equidistant spacing
    const auto vol = fs.dx[0] * fs.dy[0];

    static Matrix<Float, NX, NY, Layout::C> rhs_values{};

    // = Set initial guess to zero =================================================================
    std::array<HYPRE_Int, 2> ilower = {0, 0};
    std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
    std::fill_n(rhs_values.get_data(), rhs_values.size(), 0.0);
    HYPRE_StructVectorSetBoxValues(sol, ilower.data(), iupper.data(), rhs_values.get_data());

    // = Set right-hand side =======================================================================
    Float mean_rhs = 0.0;
    for (size_t i = 0; i < resP.extent(0); ++i) {
      for (size_t j = 0; j < resP.extent(1); ++j) {
        rhs_values[i, j] = -vol * RHO * div[i, j] / dt;
        mean_rhs += rhs_values[i, j];
      }
    }
    mean_rhs /= rhs_values.size();
    for (size_t i = 0; i < resP.extent(0); ++i) {
      for (size_t j = 0; j < resP.extent(1); ++j) {
        rhs_values[i, j] -= mean_rhs;
      }
    }
    HYPRE_StructVectorSetBoxValues(rhs, ilower.data(), iupper.data(), rhs_values.get_data());
    // for (size_t i = 0; i < resP.extent(0); ++i) {
    //   for (size_t j = 0; j < resP.extent(1); ++j) {
    //     std::array<HYPRE_Int, NDIMS> index{static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
    //     HYPRE_StructVectorSetValues(rhs, index.data(), rhs_values[i, j]);
    //   }
    // }

    // = Solve the system ==========================================================================
    Float final_residual = -1.0;
    HYPRE_Int num_iter   = -1;
    ierr                 = HYPRE_StructGMRESSolve(solver, matrix, rhs, sol);
    HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_residual);
    HYPRE_StructGMRESGetNumIterations(solver, &num_iter);

    for (size_t i = 0; i < resP.extent(0); ++i) {
      for (size_t j = 0; j < resP.extent(1); ++j) {
        std::array<HYPRE_Int, NDIMS> idx = {static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
        HYPRE_StructVectorGetValues(sol, idx.data(), &resP[i, j]);
      }
    }

    if (final_residual > 100.0 * PRESSURE_TOL) {
      if (ierr != 0) {
        HYPRE_DescribeError(ierr, buffer.data());
        Igor::Warn("Could not solve the system successfully: {}", buffer.data());
        HYPRE_ClearError(ierr);
      }

      Igor::Warn("Residual pressure correction = {}", final_residual);
      Igor::Warn("Num. iterations pressure correction = {}", num_iter);
      res = false;
    }

    return res;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
