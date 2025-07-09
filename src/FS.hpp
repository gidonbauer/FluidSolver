#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>

// #include "Config.hpp"
#include "Container.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
struct FS {
  Float visc_gas{};
  Float visc_liquid{};
  Float rho_gas{};
  Float rho_liquid{};

  Matrix<Float, NX, NY> visc{};
  // TODO: visc must be Interpolated onto the corners
  // // Matrix<Float, NX + 1, NY> visc_u_stag{};
  // Matrix<Float, NX, NY + 1> visc_v_stag{};

  Matrix<Float, NX, NY> rho{};
  Matrix<Float, NX + 1, NY> rho_u_stag{};
  Matrix<Float, NX, NY + 1> rho_v_stag{};

  Vector<Float, NX + 1> x{};
  Vector<Float, NX> xm{};
  Vector<Float, NX> dx{};

  Vector<Float, NY + 1> y{};
  Vector<Float, NY> ym{};
  Vector<Float, NY> dy{};

  Matrix<Float, NX + 1, NY> U{};
  Matrix<Float, NX + 1, NY> U_old{};

  Matrix<Float, NX, NY + 1> V{};
  Matrix<Float, NX, NY + 1> V_old{};

  Matrix<Float, NX, NY> p{};
};

// TODO: Clipped Neumann?
enum class BCond : uint8_t { DIRICHLET, NEUMANN };
enum : Index { LEFT, RIGHT, BOTTOM, TOP, NSIDES };

template <typename Float>
struct FlowBConds {
  std::array<BCond, NSIDES> types;
  std::array<Float, NSIDES> U;
  std::array<Float, NSIDES> V;
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
auto adjust_dt(const FS<Float, NX, NY>& fs, Float cfl_max, Float dt_max) -> Float {
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;
  Float CFLv_x = 0.0;
  Float CFLv_y = 0.0;

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      CFLc_x = std::max(CFLc_x, (fs.U[i, j] + fs.U[i + 1, j]) / 2 / fs.dx[i]);
      CFLc_y = std::max(CFLc_y, (fs.V[i, j] + fs.V[i, j + 1]) / 2 / fs.dy[j]);
      CFLv_x = std::max(CFLv_x, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dx[i]) * fs.rho[i, j]));
      CFLv_y = std::max(CFLv_y, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dy[j]) * fs.rho[i, j]));
    }
  }

  return std::min(cfl_max / std::max({CFLc_x, CFLc_y, CFLv_x, CFLv_y}), dt_max);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_dmomdt(const FS<Float, NX, NY>& fs,
                 Matrix<Float, NX + 1, NY>& dmomUdt,
                 Matrix<Float, NX, NY + 1>& dmomVdt) {
  // TODO: Use the correct dx and dy in case of non-uniform grids (not planned at the moment)

  // TODO: Interpolate onto the staggered mesh rho and visc, at the moment we assume that they are
  //       constant and it makes no difference but for two-phase flows this is not correct anymore
  // TODO: Use hybrid interpolation scheme: upwind in case of large density jump, centered otherwise
  static Matrix<Float, NX, NY> FX{};
  static Matrix<Float, NX, NY> FY{};
  std::fill_n(dmomUdt.get_data(), dmomUdt.size(), 0.0);
  std::fill_n(dmomVdt.get_data(), dmomVdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  const auto rho_eps = 1e-3 * std::min(fs.rho_gas, fs.rho_liquid);
  auto use_upwind    = [rho_eps](Float rho_minus, Float rho_plus) {
    return std::abs(rho_plus - rho_minus) > rho_eps;
  };
  auto hybrid_interp = [use_upwind](Float rho_minus,
                                    Float rho_plus,
                                    Float velo_minus,
                                    Float velo_plus) -> std::array<Float, 2> {
    if (!use_upwind(rho_minus, rho_plus)) {
      return {
          (rho_plus + rho_minus) / 2.0,
          (velo_plus + velo_minus) / 2.0,
      };
    }
    if (velo_plus + velo_minus >= 0.0) { return {rho_minus, velo_minus}; }
    return {rho_plus, velo_plus};
  };

  // = Calculate dmomUdt ===========================================================================
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
      //    = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
      {
        const auto [rho_hybrid, U_hybrid] =
            hybrid_interp(fs.rho_u_stag[i, j], fs.rho_u_stag[i + 1, j], fs.U[i, j], fs.U[i + 1, j]);

        FX[i, j] = -rho_hybrid * U_hybrid * ((fs.U[i + 1, j] + fs.U[i, j]) / 2) +
                   fs.visc[i, j] * (2.0 * (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] -
                                    2.0 / 3.0 *
                                        ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                                         (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
                   fs.p[i, j];
      }

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FY = -rho*U*V + mu*(dUdy + dVdx)
        const auto [rho_hybrid, U_hybrid] =
            hybrid_interp(fs.rho_u_stag[i, j], fs.rho_u_stag[i, j + 1], fs.U[i, j], fs.U[i, j + 1]);

        FY[i, j] = -rho_hybrid *                                            //
                       U_hybrid *                                           //
                       (fs.V[i - 1, j + 1] + fs.V[i, j + 1]) / 2 +          //
                   fs.visc[i, j] *                                          //
                       ((fs.U[i, j + 1] - fs.U[i, j]) / fs.dy[j] +          //
                        (fs.V[i, j + 1] - fs.V[i - 1, j + 1]) / fs.dx[i]);  //
      } else {
        // FY[i, j] = 0.0;
        // TODO: For debugging purposes, remove later.
        FY[i, j] = std::numeric_limits<Float>::quiet_NaN();
      }
    }
  }
  for (Index i = 1; i < dmomUdt.extent(0) - 1; ++i) {
    for (Index j = 1; j < dmomUdt.extent(1) - 1; ++j) {
      dmomUdt[i, j] = (FX[i, j] - FX[i - 1, j]) / fs.dx[i - 1] +  //
                      (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }

  // = Calculate dmomVdt ===========================================================================
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FX = -rho*U*V + mu*(dVdx + dUdy)
        const auto [rho_hybrid, V_hybrid] = hybrid_interp(fs.rho_v_stag[i - 1, j + 1],
                                                          fs.rho_v_stag[i, j + 1],
                                                          fs.V[i - 1, j + 1],
                                                          fs.V[i, j + 1]);

        FX[i, j] = -rho_hybrid *                                            //
                       (fs.U[i, j] + fs.U[i, j + 1]) / 2 *                  //
                       V_hybrid +                                           //
                   fs.visc[i, j] *                                          //
                       ((fs.U[i, j + 1] - fs.U[i, j]) / fs.dy[j] +          //
                        (fs.V[i, j + 1] - fs.V[i - 1, j + 1]) / fs.dx[i]);  //
      } else {
        FX[i, j] = std::numeric_limits<Float>::quiet_NaN();
        // FX[i,j] = 0.0;
      }

      // FY = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
      //    = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
      {
        const auto [rho_hybrid, V_hybrid] =
            hybrid_interp(fs.rho_v_stag[i, j], fs.rho_v_stag[i, j + 1], fs.V[i, j], fs.V[i, j + 1]);

        FY[i, j] = -rho_hybrid * V_hybrid * ((fs.V[i, j] + fs.V[i, j + 1]) / 2) +
                   fs.visc[i, j] * (2.0 * (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j] -
                                    2.0 / 3.0 *
                                        ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                                         (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
                   fs.p[i, j];
      }
    }
  }
  for (Index i = 1; i < dmomVdt.extent(0) - 1; ++i) {
    for (Index j = 1; j < dmomVdt.extent(1) - 1; ++j) {
      dmomVdt[i, j] = (FX[i + 1, j - 1] - FX[i, j - 1]) / fs.dx[i - 1] +  //
                      (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void apply_velocity_bconds(FS<Float, NX, NY>& fs, const FlowBConds<Float>& bconds) {
  // = Boundary conditions for U-component of velocity =============================================
  for (Index j = 0; j < fs.U.extent(1); ++j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET: fs.U[0, j] = bconds.U[LEFT]; break;
      case BCond::NEUMANN:   fs.U[0, j] = fs.U[1, j]; break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET: fs.U[fs.U.extent(0) - 1, j] = bconds.U[RIGHT]; break;
      case BCond::NEUMANN:   fs.U[fs.U.extent(0) - 1, j] = fs.U[fs.U.extent(0) - 2, j]; break;
    }
  }

  for (Index i = 0; i < fs.U.extent(0); ++i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET: fs.U[i, 0] = (fs.U[i, 1] + 2.0 * bconds.U[BOTTOM]) / 3.0; break;
      case BCond::NEUMANN:   fs.U[i, 0] = fs.U[i, 1]; break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET:
        fs.U[i, fs.U.extent(1) - 1] = (fs.U[i, fs.U.extent(1) - 2] + 2.0 * bconds.U[TOP]) / 3.0;
        break;
      case BCond::NEUMANN: fs.U[i, fs.U.extent(1) - 1] = fs.U[i, fs.U.extent(1) - 2]; break;
    }
  }

  // = Boundary conditions for V-component of velocity =============================================
  for (Index j = 0; j < fs.V.extent(1); ++j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET: fs.V[0, j] = (fs.V[1, j] + 2.0 * bconds.V[LEFT]) / 3.0; break;
      case BCond::NEUMANN:   fs.V[0, j] = fs.V[1, j]; break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET:
        fs.V[fs.V.extent(0) - 1, j] = (fs.V[fs.V.extent(0) - 2, j] + 2.0 * bconds.V[RIGHT]) / 3.0;
        break;
      case BCond::NEUMANN: fs.V[fs.V.extent(0) - 1, j] = fs.V[fs.V.extent(0) - 2, j]; break;
    }
  }

  for (Index i = 0; i < fs.V.extent(0); ++i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET: fs.V[i, 0] = bconds.V[BOTTOM]; break;
      case BCond::NEUMANN:   fs.V[i, 0] = fs.V[i, 1]; break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET: fs.V[i, fs.V.extent(1) - 1] = bconds.V[TOP]; break;
      case BCond::NEUMANN:   fs.V[i, fs.V.extent(1) - 1] = fs.V[i, fs.V.extent(1) - 2]; break;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr void init_mid_and_delta(FS<Float, NX, NY>& fs) noexcept {
  for (Index i = 0; i < NX; ++i) {
    fs.xm[i] = (fs.x[i] + fs.x[i + 1]) / 2;
    fs.dx[i] = fs.x[i + 1] - fs.x[i];
  }
  for (Index j = 0; j < NY; ++j) {
    fs.ym[j] = (fs.y[j] + fs.y[j + 1]) / 2;
    fs.dy[j] = fs.y[j + 1] - fs.y[j];
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr void calc_rho_and_visc(const Matrix<Float, NX, NY>& vof, FS<Float, NX, NY>& fs) noexcept {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      fs.rho[i, j]  = vof[i, j] * fs.rho_liquid + (1.0 - vof[i, j]) * fs.rho_gas;
      fs.visc[i, j] = vof[i, j] * fs.visc_liquid + (1.0 - vof[i, j]) * fs.visc_gas;
    }
  }

  for (Index i = 1; i < NX + 1 - 1; ++i) {
    for (Index j = 0; j < NY; ++j) {
      fs.rho_u_stag[i, j] = (fs.rho[i, j] + fs.rho[i - 1, j]) / 2.0;
      // fs.visc_u_stag[i, j] = (fs.visc[i, j] + fs.visc[i - 1, j]) / 2.0;
    }
  }
  for (Index j = 0; j < NY; ++j) {
    fs.rho_u_stag[0, j]  = fs.rho[0, j];
    fs.rho_u_stag[NX, j] = fs.rho[NX - 1, j];
    // fs.visc_u_stag[0, j]  = fs.visc[0, j];
    // fs.visc_u_stag[NX, j] = fs.visc[NX - 1, j];
  }

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 1; j < NY + 1 - 1; ++j) {
      fs.rho_v_stag[i, j] = (fs.rho[i, j] + fs.rho[i, j - 1]) / 2.0;
      // fs.visc_v_stag[i, j] = (fs.visc[i, j] + fs.visc[i, j - 1]) / 2.0;
    }
  }
  for (Index i = 0; i < NX; ++i) {
    fs.rho_v_stag[i, 0]  = fs.rho[i, 0];
    fs.rho_v_stag[i, NY] = fs.rho[i, NY - 1];
    // fs.visc_v_stag[i, 0]  = fs.visc[i, 0];
    // fs.visc_v_stag[i, NY] = fs.visc[i, NY - 1];
  }
}

#endif  // FLUID_SOLVER_FS_HPP_
