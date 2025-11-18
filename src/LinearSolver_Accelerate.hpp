#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_

#include <Accelerate/Accelerate.h>

#include <Igor/Math.hpp>

#include "FS.hpp"
#include "LinearSystem.hpp"
#include "Utility.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
class LinearSolver_Accelerate {
  static_assert(std::is_same_v<Float, double>, "Accelerate requires Float=double");

  PSDirichlet m_dirichlet_bc = PSDirichlet::NONE;

  Float m_tol;
  Index m_max_iter;
  bool m_is_setup = false;

  std::vector<int> m_row;
  std::vector<int> m_col;
  std::vector<double> m_mat_val;

  SparseMatrix_Double m_A{};
  DenseVector_Double m_x{};
  DenseVector_Double m_b{};

  using LS = LinearSystem<Float, NX, NY, NGHOST, Layout::C>;
  LS lin_sys{};

 public:
  constexpr LinearSolver_Accelerate(Float tol, Index max_iter, PSDirichlet dirichlet_side) noexcept
      : m_dirichlet_bc(dirichlet_side),
        m_tol(tol),
        m_max_iter(max_iter) {}

  constexpr LinearSolver_Accelerate(Float tol, Index max_iter) noexcept
      : m_tol(tol),
        m_max_iter(max_iter) {}

  // -----------------------------------------------------------------------------------------------
  constexpr LinearSolver_Accelerate(const LinearSolver_Accelerate& other) noexcept = delete;
  constexpr LinearSolver_Accelerate(LinearSolver_Accelerate&& other) noexcept      = delete;
  constexpr auto operator=(const LinearSolver_Accelerate& other) noexcept
      -> LinearSolver_Accelerate& = delete;
  constexpr auto operator=(LinearSolver_Accelerate&& other) noexcept
      -> LinearSolver_Accelerate&               = delete;
  constexpr ~LinearSolver_Accelerate() noexcept = default;

 private:
  // -----------------------------------------------------------------------------------------------
  void setup_lhs() {
    m_row.clear();
    m_col.clear();
    m_mat_val.clear();

    for_each_a(lin_sys.op, [&](Index i, Index j) {
      Index row = lin_sys.op.get_idx(i, j);
      for (size_t si = 0; si < LS::STENCIL_SIZE; ++si) {
        const auto [di, dj] = lin_sys.stencil_offsets[si];
        Index col           = lin_sys.op.get_idx(i + di, j + dj);

        if (std::abs(lin_sys.op(i, j)[si]) <= 1e-12) { continue; }
        Float val = lin_sys.op(i, j)[si];

        m_row.push_back(row);
        m_col.push_back(col);
        m_mat_val.push_back(val);
      }
    });

    SparseAttributes_t attributes{};
    const int nrows     = lin_sys.op.size();
    const int ncols     = lin_sys.op.size();
    const int nblocks   = static_cast<Index>(m_row.size());
    const int blocksize = 1;
    m_A                 = SparseConvertFromCoordinate(
        nrows, ncols, nblocks, blocksize, attributes, m_row.data(), m_col.data(), m_mat_val.data());
  }

 public:
  // -----------------------------------------------------------------------------------------------
  void set_pressure_operator(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    lin_sys.fill_pressure_operator(fs, m_dirichlet_bc);
    setup_lhs();
    m_is_setup = true;
  }

  // -----------------------------------------------------------------------------------------------
  void set_pressure_rhs(const FS<Float, NX, NY, NGHOST>& fs,
                        const Matrix<Float, NX, NY, NGHOST>& div,
                        Float dt) {
    // = Set right-hand side =======================================================================
    lin_sys.fill_pressure_rhs(fs, div, dt, m_dirichlet_bc);
    m_b = DenseVector_Double{
        .count = lin_sys.rhs.size(),
        .data  = lin_sys.rhs.get_data(),
    };
  }

  // -------------------------------------------------------------------------------------------------
  auto solve(Matrix<Float, NX, NY, NGHOST>& resP,
             Float* pressure_residual = nullptr,
             Index* num_iter          = nullptr) -> bool {
    IGOR_ASSERT(m_is_setup, "Solver has not been properly setup.");

    // = Set soluton vector to zero ================================================================
    fill(resP, 0.0);
    m_x = DenseVector_Double{
        .count = resP.size(),
        .data  = resP.get_data(),
    };

    // = Solve the system ==========================================================================
    SparseCGOptions opts{
        .reportError =
            []([[maybe_unused]] const char* message) { /*std::cerr << "ERROR: " << message;*/ },
        .maxIterations = m_max_iter,
        .atol          = m_tol,
        .rtol          = 0.0,
        .reportStatus  = nullptr,
        // .reportStatus  = [](const char* message) { std::cout << message; },
    };
    const auto precond = SparseCreatePreconditioner(SparsePreconditionerDiagScaling, m_A);
    const auto status  = SparseSolve(SparseConjugateGradient(opts), m_A, m_b, m_x, precond);

    switch (status) {
      case SparseIterativeConverged:      break;
      case SparseIterativeIllConditioned: Igor::Warn("Ill conditioned."); return false;
      case SparseIterativeInternalError:  Igor::Warn("Internal error."); return false;
      case SparseIterativeMaxIterations:  Igor::Warn("Max. iterations."); break;
      case SparseIterativeParameterError: Igor::Warn("Parameter error."); return false;
    }

    // = Get residual ==============================================================================
    if (pressure_residual != nullptr) {
      for_each_a(lin_sys.rhs, [&](Index i, Index j) { lin_sys.rhs(i, j) *= -1.0; });
      SparseMultiplyAdd(m_A, m_x, m_b);
      *pressure_residual = abs_max<true>(lin_sys.rhs);
    }
    if (num_iter != nullptr) { *num_iter = -1; }

    return true;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_
