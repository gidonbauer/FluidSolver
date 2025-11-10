#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_

#include <Accelerate/Accelerate.h>

#include <Igor/Math.hpp>

#include "FS.hpp"

enum class PSDirichlet : std::uint8_t { NONE, LEFT, RIGHT, BOTTOM, TOP };

template <typename Float, Index NX, Index NY, Index NGHOST>
class PS_Accelerate {
  static_assert(std::is_same_v<Float, double>, "Accelerate requires Float=double");

  PSDirichlet m_dirichlet_bc = PSDirichlet::NONE;

  Float m_tol;
  Index m_max_iter;
  bool m_is_setup                     = false;

  static constexpr size_t NDIMS       = 2;
  static constexpr Index STENCIL_SIZE = 5;
  Matrix<std::array<Float, STENCIL_SIZE>, NX, NY, NGHOST> m_operator{};
  enum : size_t { S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP };
  static constexpr std::array<std::array<Index, NDIMS>, STENCIL_SIZE> stencil_offsets = {
      std::array<Index, NDIMS>{0, 0},
      std::array<Index, NDIMS>{-1, 0},
      std::array<Index, NDIMS>{1, 0},
      std::array<Index, NDIMS>{0, -1},
      std::array<Index, NDIMS>{0, 1},
  };

  std::vector<int> m_row;
  std::vector<int> m_col;
  std::vector<double> m_mat_val;

  SparseMatrix_Double m_A{};
  DenseVector_Double m_x{};
  DenseVector_Double m_b{};

 public:
  constexpr PS_Accelerate(const FS<Float, NX, NY, NGHOST>& fs,
                          Float tol,
                          Index max_iter,
                          PSDirichlet dirichlet_side) noexcept
      : m_dirichlet_bc(dirichlet_side),
        m_tol(tol),
        m_max_iter(max_iter) {
    setup(fs);
  }

  constexpr PS_Accelerate(const FS<Float, NX, NY, NGHOST>& fs, Float tol, Index max_iter) noexcept
      : m_tol(tol),
        m_max_iter(max_iter) {
    setup(fs);
  }

  // -----------------------------------------------------------------------------------------------
  constexpr PS_Accelerate(const PS_Accelerate& other) noexcept                    = delete;
  constexpr PS_Accelerate(PS_Accelerate&& other) noexcept                         = delete;
  constexpr auto operator=(const PS_Accelerate& other) noexcept -> PS_Accelerate& = delete;
  constexpr auto operator=(PS_Accelerate&& other) noexcept -> PS_Accelerate&      = delete;
  constexpr ~PS_Accelerate() noexcept                                             = default;

  // -----------------------------------------------------------------------------------------------
  void setup(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    setup_operator(fs);
    setup_matrix();
    m_is_setup = true;
  }

 private:
  // -----------------------------------------------------------------------------------------------
  void setup_operator(const FS<Float, NX, NY, NGHOST>& fs) noexcept {
    const Float vol = fs.dx * fs.dy;

    for_each_a<Exec::Parallel>(m_operator, [&](Index i, Index j) {
      std::array<Float, STENCIL_SIZE>& s = m_operator(i, j);
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
          std::array<Float, STENCIL_SIZE>& s = m_operator(-NGHOST, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::RIGHT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          std::array<Float, STENCIL_SIZE>& s = m_operator(NX + NGHOST - 1, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::BOTTOM:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = m_operator(i, -NGHOST);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::TOP:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = m_operator(i, NY + NGHOST - 1);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::NONE: break;
    }
  }

  // -----------------------------------------------------------------------------------------------
  void setup_matrix() {
    m_row.clear();
    m_col.clear();
    m_mat_val.clear();

    for_each_a(m_operator, [&](Index i, Index j) {
      Index row = m_operator.get_idx(i, j);
      for (size_t si = 0; si < STENCIL_SIZE; ++si) {
        const auto [di, dj] = stencil_offsets[si];
        Index col           = m_operator.get_idx(i + di, j + dj);

        if (std::abs(m_operator(i, j)[si]) <= 1e-12) { continue; }
        Float val = m_operator(i, j)[si];

        m_row.push_back(row);
        m_col.push_back(col);
        m_mat_val.push_back(val);
      }
    });

    SparseAttributes_t attributes{};
    const int nrows     = m_operator.size();
    const int ncols     = m_operator.size();
    const int nblocks   = static_cast<Index>(m_row.size());
    const int blocksize = 1;
    m_A                 = SparseConvertFromCoordinate(
        nrows, ncols, nblocks, blocksize, attributes, m_row.data(), m_col.data(), m_mat_val.data());
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

    const auto vol = fs.dx * fs.dy;

    static Matrix<Float, NX, NY, NGHOST> rhs_values{};

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
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { rhs_values(i, NY + NGHOST - 1) = 0.0; });
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
    IGOR_ASSERT(!has_nan_or_inf(rhs_values), "NaN or inf in rhs_values");
    m_b = DenseVector_Double{
        .count = rhs_values.size(),
        .data  = rhs_values.get_data(),
    };

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
    const auto status =
        SparseSolve(SparseConjugateGradient(opts), m_A, m_b, m_x, SparsePreconditionerDiagScaling);

    switch (status) {
      case SparseIterativeConverged:      break;
      case SparseIterativeIllConditioned: Igor::Warn("Ill conditioned."); return false;
      case SparseIterativeInternalError:  Igor::Warn("Internal error."); return false;
      case SparseIterativeMaxIterations:  Igor::Warn("Max. iterations."); break;
      case SparseIterativeParameterError: Igor::Warn("Parameter error."); return false;
    }

    // = Get solution ==============================================================================
    if (pressure_residual != nullptr) {
      for_each_a(rhs_values, [&](Index i, Index j) { rhs_values(i, j) *= -1.0; });
      SparseMultiplyAdd(m_A, m_x, m_b);
      *pressure_residual = abs_max<true>(rhs_values);
    }
    if (num_iter != nullptr) { *num_iter = -1; }

    return true;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_ACCELERATE_HPP_
