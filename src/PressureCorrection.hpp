#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_HPP_

#include <omp.h>

#include <HYPRE_struct_ls.h>
#include <HYPRE_utilities.h>

#include <Igor/Math.hpp>

#include "FS.hpp"

#ifndef PS_PARALLEL_THRESHOLD
#ifdef __APPLE__
constexpr Index PS_PARALLEL_GRID_SIZE_THRESHOLD = 1000 * 1000;
#else
constexpr Index PS_PARALLEL_GRID_SIZE_THRESHOLD = 500 * 500;
#endif
#else
static_assert(std::is_convertible_v<std::remove_cvref_t<decltype(PS_PARALLEL_THRESHOLD)>, Index>,
              "PS_PARALLEL_THRESHOLD must have a value that must be convertible to Index.");
constexpr Index PS_PARALLEL_GRID_SIZE_THRESHOLD = PS_PARALLEL_THRESHOLD;
#endif

enum class PSSolver : std::uint8_t { GMRES, PCG, BiCGSTAB, SMG, PFMG };
enum class PSPrecond : std::uint8_t { SMG, PFMG, NONE };
enum class PSDirichlet : std::uint8_t { NONE, LEFT, RIGHT, BOTTOM, TOP };

template <typename Float, Index NX, Index NY, Index NGHOST>
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

  PSSolver m_solver_type     = PSSolver::GMRES;
  PSPrecond m_precond_type   = PSPrecond::SMG;
  PSDirichlet m_dirichlet_bc = PSDirichlet::NONE;

  Float m_tol;
  HYPRE_Int m_max_iter;
  bool m_is_setup = false;

 public:
  constexpr PS(const FS<Float, NX, NY, NGHOST>& fs,
               Float tol,
               HYPRE_Int max_iter,
               PSSolver solver_type,
               PSPrecond precond_type,
               PSDirichlet dirichlet_side) noexcept
      : m_solver_type(solver_type),
        m_precond_type(precond_type),
        m_dirichlet_bc(dirichlet_side),
        m_tol(tol),
        m_max_iter(max_iter) {
    HYPRE_Initialize();
    setup(fs);
  }

  constexpr PS(const FS<Float, NX, NY, NGHOST>& fs, Float tol, HYPRE_Int max_iter) noexcept
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
    switch (m_precond_type) {
      case PSPrecond::SMG:  HYPRE_StructSMGDestroy(m_precond); break;
      case PSPrecond::PFMG: HYPRE_StructPFMGDestroy(m_precond); break;
      case PSPrecond::NONE: break;
    }
    switch (m_solver_type) {
      case PSSolver::GMRES:    HYPRE_StructGMRESDestroy(m_solver); break;
      case PSSolver::PCG:      HYPRE_StructPCGDestroy(m_solver); break;
      case PSSolver::BiCGSTAB: HYPRE_StructBiCGSTABDestroy(m_solver); break;
      case PSSolver::SMG:      HYPRE_StructSMGDestroy(m_solver); break;
      case PSSolver::PFMG:     HYPRE_StructPFMGDestroy(m_solver); break;
    }
    HYPRE_StructVectorDestroy(m_sol);
    HYPRE_StructVectorDestroy(m_rhs);
    HYPRE_StructMatrixDestroy(m_matrix);
    HYPRE_StructStencilDestroy(m_stencil);
    HYPRE_StructGridDestroy(m_grid);

    m_is_setup = false;
  }

 public:
  // -----------------------------------------------------------------------------------------------
  void setup(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    if (m_is_setup) { destroy(); }

    // = Create structured grid ====================================================================
    {
      HYPRE_StructGridCreate(COMM, NDIMS, &m_grid);
      std::array<HYPRE_Int, 2> ilower = {-NGHOST, -NGHOST};
      std::array<HYPRE_Int, 2> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
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
    switch (m_solver_type) {
      case PSSolver::GMRES:
        HYPRE_StructGMRESCreate(COMM, &m_solver);
        HYPRE_StructGMRESSetTol(m_solver, m_tol);
        HYPRE_StructGMRESSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructGMRESSetPrintLevel(m_solver, 2);
        HYPRE_StructGMRESSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case PSSolver::PCG:
        HYPRE_StructPCGCreate(COMM, &m_solver);
        HYPRE_StructPCGSetTol(m_solver, m_tol);
        HYPRE_StructPCGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructPCGSetPrintLevel(m_solver, 2);
        HYPRE_StructPCGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case PSSolver::BiCGSTAB:
        HYPRE_StructBiCGSTABCreate(COMM, &m_solver);
        HYPRE_StructBiCGSTABSetTol(m_solver, m_tol);
        HYPRE_StructBiCGSTABSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructBiCGSTABSetPrintLevel(m_solver, 2);
        HYPRE_StructBiCGSTABSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case PSSolver::SMG:
        HYPRE_StructSMGCreate(COMM, &m_solver);
        HYPRE_StructSMGSetTol(m_solver, m_tol);
        HYPRE_StructSMGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructSMGSetPrintLevel(m_solver, 2);
        HYPRE_StructSMGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;

      case PSSolver::PFMG:
        HYPRE_StructPFMGCreate(COMM, &m_solver);
        HYPRE_StructPFMGSetTol(m_solver, m_tol);
        HYPRE_StructPFMGSetMaxIter(m_solver, m_max_iter);
#ifdef FS_HYPRE_VERBOSE
        HYPRE_StructPFMGSetPrintLevel(m_solver, 2);
        HYPRE_StructPFMGSetLogging(m_solver, 1);
#endif  // FS_HYPRE_VERBOSE
        break;
    }

    // = Create preconditioner =====================================================================
    HYPRE_PtrToStructSolverFcn precond_setup = nullptr;
    HYPRE_PtrToStructSolverFcn precond_solve = nullptr;
    switch (m_precond_type) {
      case PSPrecond::SMG:
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

      case PSPrecond::PFMG:
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

      case PSPrecond::NONE: break;
    }

    IGOR_ASSERT((m_precond_type != PSPrecond::NONE && precond_setup != nullptr &&
                 precond_solve != nullptr) ||
                    (m_precond_type == PSPrecond::NONE && precond_setup == nullptr &&
                     precond_solve == nullptr),
                "Did not set the preconditioner functions properly.");

    switch (m_solver_type) {
      case PSSolver::GMRES:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructGMRESSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructGMRESSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case PSSolver::PCG:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructPCGSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructPCGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case PSSolver::BiCGSTAB:
        if (precond_setup != nullptr && precond_solve != nullptr) {
          HYPRE_StructBiCGSTABSetPrecond(m_solver, precond_solve, precond_setup, m_precond);
        }
        HYPRE_StructBiCGSTABSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case PSSolver::SMG:
        IGOR_ASSERT(precond_setup == nullptr && precond_solve == nullptr,
                    "SMG solver does not support preconditioner.");
        HYPRE_StructSMGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;

      case PSSolver::PFMG:
        IGOR_ASSERT(precond_setup == nullptr && precond_solve == nullptr,
                    "PFMG solver does not support preconditioner.");
        HYPRE_StructPFMGSetup(m_solver, m_matrix, m_rhs, m_sol);
        break;
    }

    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) { Igor::Panic("An error occured in HYPRE."); }

    m_is_setup = true;
  }

 private:
  // -----------------------------------------------------------------------------------------------
  void setup_system_matrix(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    static Matrix<std::array<Float, STENCIL_SIZE>, NX, NY, NGHOST, Layout::F> stencil_values{};
    enum : size_t { S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP };
    std::array<HYPRE_Int, STENCIL_SIZE> stencil_indices{S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP};

    const Float vol = fs.dx * fs.dy;

    for_each_a<Exec::Parallel>(stencil_values, [&](Index i, Index j) {
      std::array<Float, STENCIL_SIZE>& s = stencil_values(i, j);
      std::fill(s.begin(), s.end(), 0.0);

      // = x-components ==========================================================================
      if (i == -NGHOST) {
        // On left
        s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i + 1, j));
        s[S_LEFT]   += 0.0;
        s[S_RIGHT]  += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i + 1, j));
      } else if (i == NX + NGHOST - 1) {
        // On right
        s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i, j));
        s[S_LEFT]   += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i, j));
        s[S_RIGHT]  += 0.0;
      } else {
        // In interior (x)
        s[S_CENTER] += -vol * (-1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i, j)) +
                               -1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i + 1, j)));
        s[S_LEFT]   += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i, j));
        s[S_RIGHT]  += -vol * 1.0 / (Igor::sqr(fs.dx) * fs.curr.rho_u_stag(i + 1, j));
      }

      // = y-components ==========================================================================
      if (j == -NGHOST) {
        // On bottom
        s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j + 1));
        s[S_BOTTOM] += 0.0;
        s[S_TOP]    += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j + 1));
      } else if (j == NY + NGHOST - 1) {
        // On top
        s[S_CENTER] += -vol * -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j));
        s[S_BOTTOM] += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j));
        s[S_TOP]    += 0.0;
      } else {
        // In interior (y)
        s[S_CENTER] += -vol * (-1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j)) +
                               -1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j + 1)));
        s[S_BOTTOM] += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j));
        s[S_TOP]    += -vol * 1.0 / (Igor::sqr(fs.dy) * fs.curr.rho_v_stag(i, j + 1));
      }
    });

    switch (m_dirichlet_bc) {
      case PSDirichlet::LEFT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          std::array<Float, STENCIL_SIZE>& s = stencil_values(-NGHOST, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::RIGHT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          std::array<Float, STENCIL_SIZE>& s = stencil_values(NX + NGHOST - 1, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::BOTTOM:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = stencil_values(i, -NGHOST);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::TOP:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = stencil_values(i, NX + NGHOST - 1);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::NONE: break;
    }

#if 1
    std::array<HYPRE_Int, NDIMS> ilower = {-NGHOST, -NGHOST};
    std::array<HYPRE_Int, NDIMS> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
    HYPRE_StructMatrixSetBoxValues(m_matrix,
                                   ilower.data(),
                                   iupper.data(),
                                   STENCIL_SIZE,
                                   stencil_indices.data(),
                                   stencil_values.get_data()->data());
#else
    for_each_a(stencil_values, [&](Index i, Index j) {
      std::array<HYPRE_Int, NDIMS> index{i, j};
      HYPRE_StructMatrixSetValues(m_matrix,
                                  index.data(),
                                  STENCIL_SIZE,
                                  stencil_indices.data(),
                                  stencil_values(i, j).data());
    });
#endif
  }

 public:
  // -------------------------------------------------------------------------------------------------
  auto solve(const FS<Float, NX, NY, NGHOST>& fs,
             const Matrix<Float, NX, NY, NGHOST>& div,
             Float dt,
             Matrix<Float, NX, NY, NGHOST>& resP,
             Float* pressure_residual = nullptr,
             Index* num_iter          = nullptr) -> bool {
    IGOR_ASSERT(m_is_setup, "Solver has not been properly setup.");

    int prev_num_threads = -1;
    if constexpr ((NX + 2 * NGHOST) * (NY + 2 * NGHOST) < PS_PARALLEL_GRID_SIZE_THRESHOLD) {
#pragma omp parallel
#pragma omp single
      { prev_num_threads = omp_get_num_threads(); }
      omp_set_num_threads(1);
    }

    static std::array<char, 1024UZ> buffer{};
    bool res       = true;

    const auto vol = fs.dx * fs.dy;

    static Matrix<Float, NX, NY, NGHOST, Layout::F> rhs_values{};

    // = Set initial guess to zero =================================================================
    std::array<HYPRE_Int, 2> ilower = {-NGHOST, -NGHOST};
    std::array<HYPRE_Int, 2> iupper = {NX + NGHOST - 1, NY + NGHOST - 1};
    std::fill_n(rhs_values.get_data(), rhs_values.size(), 0.0);
    HYPRE_StructVectorSetBoxValues(m_sol, ilower.data(), iupper.data(), rhs_values.get_data());

    // = Set right-hand side =======================================================================
    for_each_a(rhs_values, [&](Index i, Index j) { rhs_values(i, j) = -vol * div(i, j) / dt; });

    switch (m_dirichlet_bc) {
      case PSDirichlet::LEFT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { rhs_values(-NGHOST, j) = 0.0; });
        break;
      case PSDirichlet::RIGHT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { rhs_values(NX + NGHOST - 1, j) = 0.0; });
        break;
      case PSDirichlet::BOTTOM:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { rhs_values(i, -NGHOST) = 0.0; });
        break;
      case PSDirichlet::TOP:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { rhs_values(i, NX + NGHOST - 1) = 0.0; });
        break;
      case PSDirichlet::NONE:
        const Float mean_rhs = std::reduce(rhs_values.get_data(),
                                           rhs_values.get_data() + rhs_values.size(),
                                           Float{0},
                                           std::plus<>{}) /
                               static_cast<Float>(rhs_values.size());
        for_each_a<Exec::Parallel>(rhs_values,
                                   [&](Index i, Index j) { rhs_values(i, j) -= mean_rhs; });
        break;
    }
    HYPRE_StructVectorSetBoxValues(m_rhs, ilower.data(), iupper.data(), rhs_values.get_data());

    // = Solve the system ==========================================================================
    Float final_residual     = -1.0;
    HYPRE_Int local_num_iter = -1;

    switch (m_solver_type) {
      case PSSolver::GMRES:
        HYPRE_StructGMRESSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructGMRESGetNumIterations(m_solver, &local_num_iter);
        break;

      case PSSolver::PCG:
        HYPRE_StructPCGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructPCGGetNumIterations(m_solver, &local_num_iter);
        break;

      case PSSolver::BiCGSTAB:
        HYPRE_StructBiCGSTABSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructBiCGSTABGetNumIterations(m_solver, &local_num_iter);
        break;

      case PSSolver::SMG:
        HYPRE_StructSMGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructSMGGetNumIterations(m_solver, &local_num_iter);
        break;

      case PSSolver::PFMG:
        HYPRE_StructPFMGSolve(m_solver, m_matrix, m_rhs, m_sol);
        HYPRE_StructPFMGGetFinalRelativeResidualNorm(m_solver, &final_residual);
        HYPRE_StructPFMGGetNumIterations(m_solver, &local_num_iter);
        break;
    }

    for_each_a<Exec::Parallel>(resP, [&](Index i, Index j) {
      std::array<HYPRE_Int, NDIMS> idx = {static_cast<HYPRE_Int>(i), static_cast<HYPRE_Int>(j)};
      HYPRE_StructVectorGetValues(m_sol, idx.data(), &resP(i, j));
    });

    if (pressure_residual != nullptr) { *pressure_residual = final_residual; }
    if (num_iter != nullptr) { *num_iter = local_num_iter; }

    const HYPRE_Int error_flag = HYPRE_GetError();
    if (error_flag != 0) {
      if (HYPRE_CheckError(error_flag, HYPRE_ERROR_CONV) != 0) {
#ifndef FS_SILENCE_CONV_WARN
        Igor::Warn("HYPRE did not converge: residual = {:.6e}, #iterations = {}",
                   final_residual,
                   local_num_iter);
#endif  // FS_SILENCE_CONV_WARN
        HYPRE_ClearError(HYPRE_ERROR_CONV);
        res = false;
      } else {
        HYPRE_DescribeError(error_flag, buffer.data());
        Igor::Panic("An error occured in HYPRE: {}", buffer.data());
      }
    }

    if constexpr ((NX + 2 * NGHOST) * (NY + 2 * NGHOST) < PS_PARALLEL_GRID_SIZE_THRESHOLD) {
      omp_set_num_threads(prev_num_threads);
    }

    return res;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_HPP_
