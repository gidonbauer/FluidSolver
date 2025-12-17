#ifndef FLUID_SOLVER_LINEAR_SOLVER_STRUCT_HYPRE_HPP_
#define FLUID_SOLVER_LINEAR_SOLVER_STRUCT_HYPRE_HPP_

#include <array>

#include <omp.h>

#include <HYPRE_struct_ls.h>
#include <HYPRE_utilities.h>

#include <Igor/Math.hpp>

#include "FS.hpp"
#include "HYPREUtility.hpp"
#include "LinearSystem.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
class LinearSolver_StructHypre {
  static_assert(std::is_same_v<Float, double>, "HYPRE requires Float=double");

  static constexpr int COMM = -1;

  HYPRE_StructGrid m_grid{};
  HYPRE_StructStencil m_stencil{};
  HYPRE_StructMatrix m_matrix{};
  HYPRE_StructVector m_rhs{};
  HYPRE_StructVector m_sol{};
  HYPRE_StructSolver m_solver{};
  HYPRE_StructSolver m_precond{};

  HypreSolver m_solver_type   = HypreSolver::PCG;
  HyprePrecond m_precond_type = HyprePrecond::PFMG;
  PSDirichlet m_dirichlet_bc  = PSDirichlet::NONE;

  Float m_tol;
  HYPRE_Int m_max_iter;
  bool m_is_setup = false;

  using LS        = LinearSystem<Float, NX, NY, NGHOST, Layout::F>;
  LS lin_sys{};

 public:
  constexpr LinearSolver_StructHypre(Float tol,
                                     HYPRE_Int max_iter,
                                     HypreSolver solver_type,
                                     HyprePrecond precond_type,
                                     PSDirichlet dirichlet_side) noexcept
      : m_solver_type(solver_type),
        m_precond_type(precond_type),
        m_dirichlet_bc(dirichlet_side),
        m_tol(tol),
        m_max_iter(max_iter) {
    detail::initialize_hypre();
    initialize();
  }

  constexpr LinearSolver_StructHypre(Float tol, HYPRE_Int max_iter) noexcept
      : m_tol(tol),
        m_max_iter(max_iter) {
    detail::initialize_hypre();
    initialize();
  }

  // -----------------------------------------------------------------------------------------------
  constexpr LinearSolver_StructHypre(const LinearSolver_StructHypre& other) noexcept = delete;
  constexpr LinearSolver_StructHypre(LinearSolver_StructHypre&& other) noexcept      = delete;
  constexpr auto
  operator=(const LinearSolver_StructHypre& other) noexcept -> LinearSolver_StructHypre& = delete;
  constexpr auto
  operator=(LinearSolver_StructHypre&& other) noexcept -> LinearSolver_StructHypre& = delete;

  // -----------------------------------------------------------------------------------------------
  constexpr ~LinearSolver_StructHypre() noexcept {
    destroy();
    switch (m_solver_type) {
      case HypreSolver::GMRES:    HYPRE_StructGMRESDestroy(m_solver); break;
      case HypreSolver::PCG:      HYPRE_StructPCGDestroy(m_solver); break;
      case HypreSolver::BiCGSTAB: HYPRE_StructBiCGSTABDestroy(m_solver); break;
      case HypreSolver::SMG:      HYPRE_StructSMGDestroy(m_solver); break;
      case HypreSolver::PFMG:     HYPRE_StructPFMGDestroy(m_solver); break;
    }
    HYPRE_StructVectorDestroy(m_sol);
    HYPRE_StructVectorDestroy(m_rhs);
    HYPRE_StructMatrixDestroy(m_matrix);
    HYPRE_StructStencilDestroy(m_stencil);
    HYPRE_StructGridDestroy(m_grid);
    detail::finalize_hypre();
  }

  // -----------------------------------------------------------------------------------------------
  void set_pressure_operator(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    lin_sys.fill_pressure_operator(fs, m_dirichlet_bc);

    std::array<HYPRE_Int, LS::NDIMS> ilower = {-NGHOST, -NGHOST};
    std::array<HYPRE_Int, LS::NDIMS> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
    static_assert(sizeof(HYPRE_Int) == sizeof(Index));
    HYPRE_StructMatrixSetBoxValues(m_matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   LS::STENCIL_SIZE,
                                   reinterpret_cast<HYPRE_Int*>(lin_sys.stencil_indices.data()),
                                   lin_sys.op.get_data()->data());
    HYPRE_StructMatrixAssemble(m_matrix);

    setup_lhs();
  }

  // -----------------------------------------------------------------------------------------------
  void set_pressure_rhs(const FS<Float, NX, NY, NGHOST>& fs,
                        const Field2D<Float, NX, NY, NGHOST>& div,
                        Float dt) {
    lin_sys.fill_pressure_rhs(fs, div, dt, m_dirichlet_bc);
    std::array<HYPRE_Int, LS::NDIMS> ilower = {-NGHOST, -NGHOST};
    std::array<HYPRE_Int, LS::NDIMS> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
    HYPRE_StructVectorSetBoxValues(m_rhs, ilower.data(), iupper.data(), lin_sys.rhs.get_data());
  }

  // -----------------------------------------------------------------------------------------------
  void
  solve(Field2D<Float, NX, NY, NGHOST>& sol, Float* residual = nullptr, Index* num_iter = nullptr) {
    IGOR_ASSERT(m_is_setup, "Solver has not been properly setup.");

    FS_HYPRE_MAKE_SINGLE_THREADED_IF_NECESSARY

    // = Set initial guess to zero =================================================================
    std::array<HYPRE_Int, LS::NDIMS> ilower = {-NGHOST, -NGHOST};
    std::array<HYPRE_Int, LS::NDIMS> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
    std::fill_n(lin_sys.rhs.get_data(), lin_sys.rhs.size(), 0.0);
    HYPRE_StructVectorSetBoxValues(m_sol, ilower.data(), iupper.data(), lin_sys.rhs.get_data());

    // = Solve the system ==========================================================================
    Float final_residual     = -1.0;
    HYPRE_Int local_num_iter = -1;

    switch (m_solver_type) {
      case HypreSolver::GMRES:
        HYPRE_StructGMRESSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructGMRESGetNumIterations(m_solver, &local_num_iter);
        break;

      case HypreSolver::PCG:
        HYPRE_StructPCGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructPCGGetNumIterations(m_solver, &local_num_iter);
        break;

      case HypreSolver::BiCGSTAB:
        HYPRE_StructBiCGSTABSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructBiCGSTABGetNumIterations(m_solver, &local_num_iter);
        break;

      case HypreSolver::SMG:
        HYPRE_StructSMGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructSMGGetNumIterations(m_solver, &local_num_iter);
        break;

      case HypreSolver::PFMG:
        HYPRE_StructPFMGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructPFMGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructPFMGGetNumIterations(m_solver, &local_num_iter);
        break;
    }

    // = Get solution ==============================================================================
    HYPRE_StructVectorGetBoxValues(m_sol, ilower.data(), iupper.data(), lin_sys.rhs.get_data());
    for_each_a<Exec::Serial>(sol, [&](Index i, Index j) { sol(i, j) = lin_sys.rhs(i, j); });

    if (residual != nullptr) { *residual = final_residual; }
    if (num_iter != nullptr) { *num_iter = local_num_iter; }

    // = Check for errors ==========================================================================
    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) {
      if (HYPRE_CheckError(error_flag, HYPRE_ERROR_CONV) != 0) {
#ifndef FS_SILENCE_CONV_WARN
        Igor::Warn("HYPRE did not converge: residual = {:.6e}, #iterations = {}",
                   final_residual,
                   local_num_iter);
#endif  // FS_SILENCE_CONV_WARN
        HYPRE_ClearError(HYPRE_ERROR_CONV);
      } else {
        static std::array<char, HYPRE_MAX_MSG_LEN> msg_buffer{};
        HYPRE_DescribeError(error_flag, msg_buffer.data());
        Igor::Panic("An error occured in HYPRE: {}", msg_buffer.data());
      }
    }

    FS_HYPRE_RESET_THREADING
  }

 private:
  // -----------------------------------------------------------------------------------------------
  void setup_lhs() noexcept {
    FS_HYPRE_MAKE_SINGLE_THREADED_IF_NECESSARY

    if (m_is_setup) { destroy(); }

    // = Create preconditioner =====================================================================
    HYPRE_PtrToStructSolverFcn precond_setup = nullptr;
    HYPRE_PtrToStructSolverFcn precond_solve = nullptr;
    switch (m_precond_type) {
      case HyprePrecond::SMG:
        HYPRE_StructSMGCreate(COMM, &m_precond);
        HYPRE_StructSMGSetMaxIter(m_precond, 1);
        HYPRE_StructSMGSetTol(m_precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(m_precond);
        HYPRE_StructSMGSetNumPreRelax(m_precond, 1);
        HYPRE_StructSMGSetNumPostRelax(m_precond, 1);
        // HYPRE_StructSMGSetMemoryUse(m_precond, 0);
        precond_setup = HYPRE_StructSMGSetup;
        precond_solve = HYPRE_StructSMGSolve;
        break;

      case HyprePrecond::PFMG:
        {
          HYPRE_StructPFMGCreate(COMM, &m_precond);
          HYPRE_StructPFMGSetMaxIter(m_precond, 1);
#ifndef PS_PFMG_MAX_LEVELS
          constexpr HYPRE_Int MAX_LEVELS = 16;
#else
          static_assert(
              std::is_convertible_v<std::remove_cvref_t<decltype(PS_PFMG_MAX_LEVELS)>, HYPRE_Int>,
              "PS_PFMG_MAX_LEVELS must have a value that must be convertible to HYPRE_Int.");
          constexpr HYPRE_Int MAX_LEVELS = PS_PFMG_MAX_LEVELS;
#endif
          HYPRE_StructPFMGSetMaxLevels(m_precond, MAX_LEVELS);
          HYPRE_StructPFMGSetRAPType(m_precond, 1);
          HYPRE_StructPFMGSetTol(m_precond, 0.0);
          HYPRE_StructPFMGSetZeroGuess(m_precond);
          HYPRE_StructPFMGSetNumPreRelax(m_precond, 1);
          HYPRE_StructPFMGSetNumPostRelax(m_precond, 1);
          precond_setup = HYPRE_StructPFMGSetup;
          precond_solve = HYPRE_StructPFMGSolve;
        }
        break;

      case HyprePrecond::NONE: break;
    }

    IGOR_ASSERT((m_precond_type != HyprePrecond::NONE && precond_setup != nullptr &&
                 precond_solve != nullptr) ||
                    (m_precond_type == HyprePrecond::NONE && precond_setup == nullptr &&
                     precond_solve == nullptr),
                "Did not set the preconditioner functions properly.");

    switch (m_solver_type) {
      case HypreSolver::GMRES:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructGMRESSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructGMRESSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case HypreSolver::PCG:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructPCGSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructPCGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case HypreSolver::BiCGSTAB:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructBiCGSTABSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructBiCGSTABSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case HypreSolver::SMG:
        IGOR_ASSERT(precond_setup == nullptr && precond_solve == nullptr,
                    "SMG solver does not support preconditioner.");
        HYPRE_StructSMGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case HypreSolver::PFMG:
        IGOR_ASSERT(precond_setup == nullptr && precond_solve == nullptr,
                    "PFMG solver does not support preconditioner.");
        HYPRE_StructPFMGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;
    }

    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) { Igor::Panic("An error occured in HYPRE."); }

    m_is_setup = true;

    FS_HYPRE_RESET_THREADING
  }

  // -----------------------------------------------------------------------------------------------
  void initialize() noexcept {
    // = Create structured grid ====================================================================
    {
      HYPRE_StructGridCreate(COMM, LS::NDIMS, &m_grid);
      std::array<HYPRE_Int, 2> ilower = {-NGHOST, -NGHOST};
      std::array<HYPRE_Int, 2> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
      HYPRE_StructGridSetExtents(m_grid, ilower.data(), iupper.data());
      HYPRE_StructGridAssemble(m_grid);
    }

    // = Create stencil ============================================================================
    HYPRE_StructStencilCreate(LS::NDIMS, LS::STENCIL_SIZE, &m_stencil);
    for (HYPRE_Int i = 0; i < LS::STENCIL_SIZE; ++i) {
      static_assert(sizeof(HYPRE_Int) == sizeof(Index));
      auto offset = lin_sys.stencil_offsets[static_cast<size_t>(i)];
      HYPRE_StructStencilSetElement(m_stencil, i, reinterpret_cast<HYPRE_Int*>(offset.data()));
    }

    // = Create struct matrix ======================================================================
    HYPRE_StructMatrixCreate(COMM, m_grid, m_stencil, &m_matrix);
    HYPRE_StructMatrixInitialize(m_matrix);

    // = Create right-hand side ====================================================================
    HYPRE_StructVectorCreate(COMM, m_grid, &m_rhs);
    HYPRE_StructVectorInitialize(m_rhs);

    // = Create solution vector ====================================================================
    HYPRE_StructVectorCreate(COMM, m_grid, &m_sol);
    HYPRE_StructVectorInitialize(m_sol);

    // = Create solver =============================================================================
    switch (m_solver_type) {
      case HypreSolver::GMRES:
        HYPRE_StructGMRESCreate(COMM, &m_solver);
        HYPRE_StructGMRESSetTol(m_solver, m_tol);
        HYPRE_StructGMRESSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructGMRESSetPrintLevel(m_solver, 2);
        HYPRE_StructGMRESSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case HypreSolver::PCG:
        HYPRE_StructPCGCreate(COMM, &m_solver);
        HYPRE_StructPCGSetTol(m_solver, m_tol);
        HYPRE_StructPCGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructPCGSetPrintLevel(m_solver, 2);
        HYPRE_StructPCGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case HypreSolver::BiCGSTAB:
        HYPRE_StructBiCGSTABCreate(COMM, &m_solver);
        HYPRE_StructBiCGSTABSetTol(m_solver, m_tol);
        HYPRE_StructBiCGSTABSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructBiCGSTABSetPrintLevel(m_solver, 2);
        HYPRE_StructBiCGSTABSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case HypreSolver::SMG:
        HYPRE_StructSMGCreate(COMM, &m_solver);
        HYPRE_StructSMGSetTol(m_solver, m_tol);
        HYPRE_StructSMGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructSMGSetPrintLevel(m_solver, 2);
        HYPRE_StructSMGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case HypreSolver::PFMG:
        HYPRE_StructPFMGCreate(COMM, &m_solver);
        HYPRE_StructPFMGSetTol(m_solver, m_tol);
        HYPRE_StructPFMGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructPFMGSetPrintLevel(m_solver, 2);
        HYPRE_StructPFMGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;
    }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void destroy() noexcept {
    switch (m_precond_type) {
      case HyprePrecond::SMG:  HYPRE_StructSMGDestroy(m_precond); break;
      case HyprePrecond::PFMG: HYPRE_StructPFMGDestroy(m_precond); break;
      case HyprePrecond::NONE: break;
    }

    m_is_setup = false;
  }
};

#endif  // FLUID_SOLVER_LINEAR_SOLVER_STRUCT_HYPRE_HPP_
