#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "PressureCorrection_Accelerate.hpp"

// = Setup =========================================================================================
using Float                 = double;

constexpr Index NGHOST      = 1;

constexpr Float X_MIN       = 0.0;
constexpr Float X_MAX       = 5.0;
constexpr Float Y_MIN       = 0.0;
constexpr Float Y_MAX       = 1.0;

constexpr Float RHO         = 0.5;
constexpr Float VISC        = 1e-3;

constexpr Float PS_TOL      = 1e-6;
constexpr Index PS_MAX_ITER = 10'000;

constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = 1.0, .V = 0.0},
    .right  = Neumann{},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Setup =========================================================================================

template <Index NX, Index NY>
void run_bench() {
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas    = VISC,
      .visc_liquid = VISC,
      .rho_gas     = RHO,
      .rho_liquid  = RHO,
  };
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  apply_velocity_bconds(fs, bconds);

  Matrix<Float, NX, NY, NGHOST> div{};
  Matrix<Float, NX, NY, NGHOST> resP{};
  Float dt = 1e-2;

  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

  IGOR_TIME_SCOPE(Igor::detail::format("HYPRE PCG-PFMG for {}x{}", NX, NY)) {
    Index num_iter;
    Float residual;

    PS ps(fs, PS_TOL, PS_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::NONE);
    ps.solve(fs, div, dt, resP, &residual, &num_iter);
    Igor::Info("HYPRE: residual = {:.6e}", residual);
    Igor::Info("HYPRE: num_iter = {}", num_iter);
  }
  std::cout << "------------------------------------------------------------\n";

  IGOR_TIME_SCOPE(Igor::detail::format("HYPRE for PCG-SMG {}x{}", NX, NY)) {
    Index num_iter;
    Float residual;

    PS ps(fs, PS_TOL, PS_MAX_ITER, PSSolver::PCG, PSPrecond::SMG, PSDirichlet::NONE);
    ps.solve(fs, div, dt, resP, &residual, &num_iter);
    Igor::Info("HYPRE: residual = {:.6e}", residual);
    Igor::Info("HYPRE: num_iter = {}", num_iter);
  }
  std::cout << "------------------------------------------------------------\n";

  IGOR_TIME_SCOPE(Igor::detail::format("HYPRE for BiCGSTAB-PFMG {}x{}", NX, NY)) {
    Index num_iter;
    Float residual;

    PS ps(fs, PS_TOL, PS_MAX_ITER, PSSolver::BiCGSTAB, PSPrecond::PFMG, PSDirichlet::NONE);
    ps.solve(fs, div, dt, resP, &residual, &num_iter);
    Igor::Info("HYPRE: residual = {:.6e}", residual);
    Igor::Info("HYPRE: num_iter = {}", num_iter);
  }
  std::cout << "------------------------------------------------------------\n";

  IGOR_TIME_SCOPE(Igor::detail::format("HYPRE for BiCGSTAB-SMG {}x{}", NX, NY)) {
    Index num_iter;
    Float residual;

    PS ps(fs, PS_TOL, PS_MAX_ITER, PSSolver::BiCGSTAB, PSPrecond::SMG, PSDirichlet::NONE);
    ps.solve(fs, div, dt, resP, &residual, &num_iter);
    Igor::Info("HYPRE: residual = {:.6e}", residual);
    Igor::Info("HYPRE: num_iter = {}", num_iter);
  }
  std::cout << "------------------------------------------------------------\n";

  IGOR_TIME_SCOPE(Igor::detail::format("Apple Accelerate for {}x{}", NX, NY)) {
    Index num_iter;
    Float residual;

    PS_Accelerate psa(fs, PS_TOL, PS_MAX_ITER);
    psa.solve(fs, div, dt, resP, &residual, &num_iter);
    Igor::Info("Apple Accelerate: residual = {:.6e}", residual);
    Igor::Info("Apple Accelerate: num_iter = {}", num_iter);
  }
  std::cout << "--------------------------------------------------------------------------------\n";
}

auto main() -> int {
  run_bench<5 * 32, 32>();
  run_bench<5 * 128, 128>();
  run_bench<256, 512>();
  // run_bench<32, 2048>();
}
