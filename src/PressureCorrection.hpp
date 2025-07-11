#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_HPP_

#include <HYPRE_struct_ls.h>
#include <HYPRE_utilities.h>

#include <Igor/Math.hpp>

#include "FS.hpp"

template <typename Float, Index NX, Index NY>
class PS {
  static_assert(std::is_same_v<Float, double>, "HYPRE requires Float=double");

  static constexpr int COMM               = -1;
  static constexpr size_t NDIMS           = 2;
  static constexpr HYPRE_Int STENCIL_SIZE = 5;

  HYPRE_StructGrid m_grid{};
  HYPRE_StructStencil m_stencil{};
  HYPRE_StructMatrix m_matrix{};
  HYPRE_StructVector m_rhs{};
  HYPRE_StructVector m_sol{};
  HYPRE_StructSolver m_solver{};
  HYPRE_StructSolver m_precond{};

  Float m_tol;

 public:
  // TODO: Assumes equidistant spacing in x- and y-direction respectively
  constexpr PS(const FS<Float, NX, NY>& fs, Float tol, HYPRE_Int max_iter) noexcept
      : m_tol(tol) {
    HYPRE_Initialize();

    // = Create structured grid ====================================================================
    {
      HYPRE_StructGridCreate(COMM, NDIMS, &m_grid);
      std::array<HYPRE_Int, 2> ilower = {0, 0};
      std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
      HYPRE_StructGridSetExtents(m_grid, ilower.data(), iupper.data());
      HYPRE_StructGridAssemble(m_grid);
    }

    // = Create stencil ============================================================================
    std::array<std::array<HYPRE_Int, NDIMS>, STENCIL_SIZE> stencil_offsets = {
        std::array<HYPRE_Int, NDIMS>{0, 0},
        std::array<HYPRE_Int, NDIMS>{-1, 0},
        std::array<HYPRE_Int, NDIMS>{1, 0},
        std::array<HYPRE_Int, NDIMS>{0, -1},
        std::array<HYPRE_Int, NDIMS>{0, 1},
    };
    HYPRE_StructStencilCreate(NDIMS, STENCIL_SIZE, &m_stencil);
    for (HYPRE_Int i = 0; i < STENCIL_SIZE; ++i) {
      HYPRE_StructStencilSetElement(
          m_stencil, i, stencil_offsets[static_cast<size_t>(i)].data());  // NOLINT
    }

    // = Setup struct matrix =======================================================================
    HYPRE_StructMatrixCreate(COMM, m_grid, m_stencil, &m_matrix);
    HYPRE_StructMatrixInitialize(m_matrix);
    setup_system_matrix(fs);
    HYPRE_StructMatrixAssemble(m_matrix);

    // = Create right-hand side ====================================================================
    HYPRE_StructVectorCreate(COMM, m_grid, &m_rhs);
    HYPRE_StructVectorInitialize(m_rhs);

    // = Create solution vector ====================================================================
    HYPRE_StructVectorCreate(COMM, m_grid, &m_sol);
    HYPRE_StructVectorInitialize(m_sol);

    // = Create solver =============================================================================
    HYPRE_StructGMRESCreate(COMM, &m_solver);
    HYPRE_StructGMRESSetTol(m_solver, tol);
    HYPRE_StructGMRESSetMaxIter(m_solver, max_iter);
#ifdef FS_HYPRE_VERBOSE
    HYPRE_StructGMRESSetPrintLevel(m_solver, 2);
    HYPRE_StructGMRESSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE

    // = Create preconditioner =====================================================================
    HYPRE_StructSMGCreate(COMM, &m_precond);
    HYPRE_StructSMGSetMaxIter(m_precond, 1);
    HYPRE_StructSMGSetTol(m_precond, 0.0);
    HYPRE_StructSMGSetZeroGuess(m_precond);
    HYPRE_StructSMGSetNumPreRelax(m_precond, 1);
    HYPRE_StructSMGSetNumPostRelax(m_precond, 1);
    HYPRE_StructSMGSetMemoryUse(m_precond, 0);
    HYPRE_StructGMRESSetPrecond(m_solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, m_precond);
    HYPRE_StructGMRESSetup(m_solver, m_matrix, m_rhs, m_sol);

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
    HYPRE_StructSMGDestroy(m_precond);
    HYPRE_StructGMRESDestroy(m_solver);
    HYPRE_StructVectorDestroy(m_sol);
    HYPRE_StructVectorDestroy(m_rhs);
    HYPRE_StructMatrixDestroy(m_matrix);
    HYPRE_StructStencilDestroy(m_stencil);
    HYPRE_StructGridDestroy(m_grid);
    HYPRE_Finalize();
  }

  void setup_system_matrix(const FS<Float, NX, NY>& fs) noexcept {
    // TODO: Divide stencil by rho on staggered mesh

    // TODO: Assumes equidistant spacing in x- and y-direction respectively
    const auto dx  = fs.dx[0];
    const auto dy  = fs.dy[0];
    const auto vol = dx * dy;

    // Interior points
    {
      std::vector<double> matrix_values(static_cast<size_t>((NX - 2) * (NY - 2) * STENCIL_SIZE));
      std::array<HYPRE_Int, STENCIL_SIZE> stencil_indices = {0, 1, 2, 3, 4};
      for (size_t i = 0; i < matrix_values.size(); i += STENCIL_SIZE) {
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
      HYPRE_StructMatrixSetBoxValues(m_matrix,
                                     ilower.data(),
                                     iupper.data(),
                                     STENCIL_SIZE,
                                     stencil_indices.data(),
                                     matrix_values.data());
    }

    // Boundary points
    // Left:
    {
      std::array<HYPRE_Int, NDIMS> ilower = {0, 1};
      std::array<HYPRE_Int, NDIMS> iupper = {0, NY - 2};
      std::vector<HYPRE_Int> stencil_indices(STENCIL_SIZE * (NY - 2));
      std::vector<double> values(stencil_indices.size());
      for (size_t i = 0; i < stencil_indices.size(); i += STENCIL_SIZE) {
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
      HYPRE_StructMatrixSetBoxValues(m_matrix,
                                     ilower.data(),
                                     iupper.data(),
                                     STENCIL_SIZE,
                                     stencil_indices.data(),
                                     values.data());
    }
    // Right:
    {
      std::array<HYPRE_Int, NDIMS> ilower = {NX - 1, 1};
      std::array<HYPRE_Int, NDIMS> iupper = {NX - 1, NY - 2};
      std::vector<HYPRE_Int> stencil_indices(STENCIL_SIZE * (NY - 2));
      std::vector<double> values(stencil_indices.size());
      for (size_t i = 0; i < stencil_indices.size(); i += STENCIL_SIZE) {
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
      HYPRE_StructMatrixSetBoxValues(m_matrix,
                                     ilower.data(),
                                     iupper.data(),
                                     STENCIL_SIZE,
                                     stencil_indices.data(),
                                     values.data());
    }
    // Bottom:
    {
      std::array<HYPRE_Int, NDIMS> ilower = {1, 0};
      std::array<HYPRE_Int, NDIMS> iupper = {NX - 2, 0};
      std::vector<HYPRE_Int> stencil_indices(STENCIL_SIZE * (NX - 2));
      std::vector<double> values(stencil_indices.size());
      for (size_t i = 0; i < stencil_indices.size(); i += STENCIL_SIZE) {
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
      HYPRE_StructMatrixSetBoxValues(m_matrix,
                                     ilower.data(),
                                     iupper.data(),
                                     STENCIL_SIZE,
                                     stencil_indices.data(),
                                     values.data());
    }
    // Top:
    {
      std::array<HYPRE_Int, NDIMS> ilower = {1, NY - 1};
      std::array<HYPRE_Int, NDIMS> iupper = {NX - 2, NY - 1};
      std::vector<HYPRE_Int> stencil_indices(STENCIL_SIZE * (NX - 2));
      std::vector<double> values(stencil_indices.size());
      for (size_t i = 0; i < stencil_indices.size(); i += STENCIL_SIZE) {
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
      HYPRE_StructMatrixSetBoxValues(m_matrix,
                                     ilower.data(),
                                     iupper.data(),
                                     STENCIL_SIZE,
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
          m_matrix, idx.data(), STENCIL_SIZE, stencil_indices.data(), values.data());
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
          m_matrix, idx.data(), STENCIL_SIZE, stencil_indices.data(), values.data());
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
          m_matrix, idx.data(), STENCIL_SIZE, stencil_indices.data(), values.data());
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
          m_matrix, idx.data(), STENCIL_SIZE, stencil_indices.data(), values.data());
    }
  }

  // -------------------------------------------------------------------------------------------------
  [[nodiscard]] auto solve(const FS<Float, NX, NY>& fs,
                           const Matrix<Float, NX, NY>& div,
                           Float dt,
                           Matrix<Float, NX, NY>& resP,
                           Float* pressure_residual = nullptr,
                           Index* num_iter          = nullptr) -> bool {
    static std::array<char, 1024UZ> buffer{};
    HYPRE_Int ierr = 0;
    bool res       = true;

    // TODO: Assumes equidistant spacing
    const auto vol = fs.dx[0] * fs.dy[0];

    static Matrix<Float, NX, NY, Layout::F> rhs_values{};

    // = Set initial guess to zero =================================================================
    std::array<HYPRE_Int, 2> ilower = {0, 0};
    std::array<HYPRE_Int, 2> iupper = {NX - 1, NY - 1};
    std::fill_n(rhs_values.get_data(), rhs_values.size(), 0.0);
    HYPRE_StructVectorSetBoxValues(m_sol, ilower.data(), iupper.data(), rhs_values.get_data());

    // = Set right-hand side =======================================================================
    Float mean_rhs = 0.0;
    for (Index i = 0; i < resP.extent(0); ++i) {
      for (Index j = 0; j < resP.extent(1); ++j) {
        // TODO: Remove rho here and divide lhs stencil by the approproate rho on the staggered mesh
        rhs_values[i, j] = -vol * fs.rho[i, j] * div[i, j] / dt;
        mean_rhs += rhs_values[i, j];
      }
    }
    mean_rhs /= static_cast<Float>(rhs_values.size());
    for (Index i = 0; i < resP.extent(0); ++i) {
      for (Index j = 0; j < resP.extent(1); ++j) {
        rhs_values[i, j] -= mean_rhs;
      }
    }
    HYPRE_StructVectorSetBoxValues(m_rhs, ilower.data(), iupper.data(), rhs_values.get_data());

    // = Solve the system ==========================================================================
    Float final_residual     = -1.0;
    HYPRE_Int local_num_iter = -1;
    ierr                     = HYPRE_StructGMRESSolve(m_solver, m_matrix, m_rhs, m_sol);
    HYPRE_StructGMRESGetFinalRelativeResidualNorm(m_solver, &final_residual);
    HYPRE_StructGMRESGetNumIterations(m_solver, &local_num_iter);

    for (Index i = 0; i < resP.extent(0); ++i) {
      for (Index j = 0; j < resP.extent(1); ++j) {
        std::array<HYPRE_Int, NDIMS> idx = {static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
        HYPRE_StructVectorGetValues(m_sol, idx.data(), &resP[i, j]);
      }
    }

    if (final_residual > 100.0 * m_tol) {
      if (ierr != 0) {
        HYPRE_DescribeError(ierr, buffer.data());
        Igor::Warn("Could not solve the system successfully: {}", buffer.data());
        HYPRE_ClearError(ierr);
      }

      Igor::Warn("Residual pressure correction = {}", final_residual);
      Igor::Warn("Num. iterations pressure correction = {}", local_num_iter);
      res = false;
    }
    if (pressure_residual != nullptr) { *pressure_residual = final_residual; }
    if (num_iter != nullptr) { *num_iter = local_num_iter; }

    return res;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
