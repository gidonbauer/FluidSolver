#ifndef FLUID_SOLVER_LINEAR_SYSTEM_HPP_
#define FLUID_SOLVER_LINEAR_SYSTEM_HPP_

#include <array>

#include <Igor/Math.hpp>

#include "FS.hpp"

enum class PSDirichlet : std::uint8_t { NONE, LEFT, RIGHT, BOTTOM, TOP };

template <typename Float, Index NX, Index NY, Index NGHOST, Layout LAYOUT>
class LinearSystem {
 public:
  static constexpr size_t NDIMS       = 2;
  static constexpr Index STENCIL_SIZE = 5;  // TODO: Is this enough for every case?
  enum : size_t { S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP };

  std::array<std::array<Index, NDIMS>, STENCIL_SIZE> stencil_offsets = {
      std::array<Index, NDIMS>{0, 0},
      std::array<Index, NDIMS>{-1, 0},
      std::array<Index, NDIMS>{1, 0},
      std::array<Index, NDIMS>{0, -1},
      std::array<Index, NDIMS>{0, 1},
  };
  std::array<Index, STENCIL_SIZE> stencil_indices{S_CENTER, S_LEFT, S_RIGHT, S_BOTTOM, S_TOP};

  Matrix<std::array<Float, STENCIL_SIZE>, NX, NY, NGHOST, LAYOUT> op{};
  Matrix<Float, NX, NY, NGHOST, LAYOUT> rhs{};

  // -----------------------------------------------------------------------------------------------
  void fill_pressure_operator(const FS<Float, NX, NY, NGHOST>& fs,
                              PSDirichlet dirichlet_bc = PSDirichlet::NONE) noexcept {
    const Float vol = fs.dx * fs.dy;

    for_each_a<Exec::Parallel>(op, [&](Index i, Index j) {
      std::array<Float, STENCIL_SIZE>& s = op(i, j);
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

    switch (dirichlet_bc) {
      case PSDirichlet::LEFT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          std::array<Float, STENCIL_SIZE>& s = op(-NGHOST, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::RIGHT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          std::array<Float, STENCIL_SIZE>& s = op(NX + NGHOST - 1, j);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::BOTTOM:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = op(i, -NGHOST);
          s[S_CENTER]                        = 1.0;
          s[S_LEFT]                          = 0.0;
          s[S_RIGHT]                         = 0.0;
          s[S_BOTTOM]                        = 0.0;
          s[S_TOP]                           = 0.0;
        });
        break;
      case PSDirichlet::TOP:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          std::array<Float, STENCIL_SIZE>& s = op(i, NY + NGHOST - 1);
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
  void fill_pressure_rhs(const FS<Float, NX, NY, NGHOST>& fs,
                         const Matrix<Float, NX, NY, NGHOST>& div,
                         Float dt,
                         PSDirichlet dirichlet_bc = PSDirichlet::NONE) noexcept {
    const auto vol = fs.dx * fs.dy;

    // = Set right-hand side =======================================================================
    for_each_a(rhs, [&](Index i, Index j) { rhs(i, j) = -vol * div(i, j) / dt; });

    switch (dirichlet_bc) {
      case PSDirichlet::LEFT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { rhs(-NGHOST, j) = 0.0; });
        break;
      case PSDirichlet::RIGHT:
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { rhs(NX + NGHOST - 1, j) = 0.0; });
        break;
      case PSDirichlet::BOTTOM:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { rhs(i, -NGHOST) = 0.0; });
        break;
      case PSDirichlet::TOP:
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { rhs(i, NY + NGHOST - 1) = 0.0; });
        break;
      case PSDirichlet::NONE:
        const Float mean_rhs =
            std::reduce(rhs.get_data(), rhs.get_data() + rhs.size(), Float{0}, std::plus<>{}) /
            static_cast<Float>(rhs.size());
        for_each_a<Exec::Parallel>(rhs, [&](Index i, Index j) { rhs(i, j) -= mean_rhs; });
        break;
    }
    IGOR_ASSERT(!has_nan_or_inf(rhs), "NaN or inf in rhs_values");
  }
};

#endif  // FLUID_SOLVER_LINEAR_SYSTEM_HPP_
