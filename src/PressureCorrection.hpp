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
  HYPRE_Int m_max_iter;
  bool m_is_setup = false;

 public:
  constexpr PS(const FS<Float, NX, NY>& fs, Float tol, HYPRE_Int max_iter) noexcept
      : m_tol(tol),
        m_max_iter(max_iter) {
    HYPRE_Initialize();
    setup(fs);
  }

  // -----------------------------------------------------------------------------------------------
  constexpr PS(const PS& other) noexcept                    = delete;
  constexpr PS(PS&& other) noexcept                         = delete;
  constexpr auto operator=(const PS& other) noexcept -> PS& = delete;
  constexpr auto operator=(PS&& other) noexcept -> PS&      = delete;

  // -----------------------------------------------------------------------------------------------
  constexpr ~PS() noexcept {
    destroy();
    HYPRE_Finalize();
  }

 private:
  // -----------------------------------------------------------------------------------------------
  constexpr void destroy() noexcept {
    HYPRE_StructSMGDestroy(m_precond);
    HYPRE_StructGMRESDestroy(m_solver);
    HYPRE_StructVectorDestroy(m_sol);
    HYPRE_StructVectorDestroy(m_rhs);
    HYPRE_StructMatrixDestroy(m_matrix);
    HYPRE_StructStencilDestroy(m_stencil);
    HYPRE_StructGridDestroy(m_grid);

    m_is_setup = false;
  }

 public:
  // -----------------------------------------------------------------------------------------------
  void setup(const FS<Float, NX, NY>& fs) noexcept {
    if (m_is_setup) { destroy(); }

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
    HYPRE_StructGMRESSetTol(m_solver, m_tol);
    HYPRE_StructGMRESSetMaxIter(m_solver, m_max_iter);
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

    m_is_setup = true;
  }

 private:
  // -----------------------------------------------------------------------------------------------
  void setup_system_matrix(const FS<Float, NX, NY>& fs) noexcept {
    static Matrix<std::array<Float, STENCIL_SIZE>, NX, NY, Layout::F> stencil_values{};
    enum : size_t { S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP };
    std::array<HYPRE_Int, STENCIL_SIZE> stencil_indices{S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP};

    const Float vol = fs.dx * fs.dy;

    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        std::array<Float, STENCIL_SIZE>& s = stencil_values[i, j];
        std::fill(s.begin(), s.end(), 0.0);

        // = x-components ==========================================================================
        if (i == 0) {
          // On left
          s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i + 1, j]);
          s[S_LEFT]   += 0.0;
          s[S_RIGHT]  += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i + 1, j]);
        } else if (i == NX - 1) {
          // On right
          s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i, j]);
          s[S_LEFT]   += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i, j]);
          s[S_RIGHT]  += 0.0;
        } else {
          // In interior (x)
          s[S_CENTER] += -vol * (-1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i, j]) +
                                 -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i + 1, j]));
          s[S_LEFT]   += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i, j]);
          s[S_RIGHT]  += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag[i + 1, j]);
        }

        // = y-components ==========================================================================
        if (j == 0) {
          // On bottom
          s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j + 1]);
          s[S_BOTTOM] += 0.0;
          s[S_TOP]    += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j + 1]);
        } else if (j == NY - 1) {
          // On top
          s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j]);
          s[S_BOTTOM] += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j]);
          s[S_TOP]    += 0.0;
        } else {
          // In interior (y)
          s[S_CENTER] += -vol * (-1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j]) +
                                 -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j + 1]));
          s[S_BOTTOM] += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j]);
          s[S_TOP]    += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag[i, j + 1]);
        }
      }
    }

#if 1
    for (Index i = 0; i < NX; ++i) {
      for (Index j = 0; j < NY; ++j) {
        std::array<HYPRE_Int, NDIMS> index{i, j};
        HYPRE_StructMatrixSetValues(m_matrix,
                                    index.data(),
                                    STENCIL_SIZE,
                                    stencil_indices.data(),
                                    stencil_values[i, j].data());
      }
    }
#else
    std::array<HYPRE_Int, NDIMS> ilower = {0, 0};
    std::array<HYPRE_Int, NDIMS> iupper = {NX - 1, NY - 1};
    HYPRE_StructMatrixSetBoxValues(m_matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   STENCIL_SIZE,
                                   stencil_indices.data(),
                                   stencil_values.get_data()->data());
#endif
  }

 public:
  // -------------------------------------------------------------------------------------------------
  [[nodiscard]] auto solve(const FS<Float, NX, NY>& fs,
                           const Matrix<Float, NX, NY>& div,
                           Float dt,
                           Matrix<Float, NX, NY>& resP,
                           Float* pressure_residual = nullptr,
                           Index* num_iter          = nullptr) -> bool {
    IGOR_ASSERT(m_is_setup, "Solver has not been properly setup.");

    static std::array<char, 1024UZ> buffer{};
    bool res       = true;

    const auto vol = fs.dx * fs.dy;

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
        rhs_values[i, j]  = -vol * div[i, j] / dt;
        mean_rhs         += rhs_values[i, j];
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
    HYPRE_StructGMRESSolve(m_solver, m_matrix, m_rhs, m_sol);
    HYPRE_StructGMRESGetFinalRelativeResidualNorm(m_solver, &final_residual);
    HYPRE_StructGMRESGetNumIterations(m_solver, &local_num_iter);

    for (Index i = 0; i < resP.extent(0); ++i) {
      for (Index j = 0; j < resP.extent(1); ++j) {
        std::array<HYPRE_Int, NDIMS> idx = {static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
        HYPRE_StructVectorGetValues(m_sol, idx.data(), &resP[i, j]);
      }
    }

    if (pressure_residual != nullptr) { *pressure_residual = final_residual; }
    if (num_iter != nullptr) { *num_iter = local_num_iter; }

    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) {
      if (HYPRE_CheckError(error_flag, HYPRE_ERROR_CONV) != 0) {
        Igor::Warn("HYPRE did not converge: residual = {:.6e}, #iterations = {}",
                   final_residual,
                   local_num_iter);
        HYPRE_ClearError(HYPRE_ERROR_CONV);
        res = false;
      } else {
        HYPRE_DescribeError(error_flag, buffer.data());
        Igor::Panic("An error occured in HYPRE: {}", buffer.data());
      }
    }

    return res;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
