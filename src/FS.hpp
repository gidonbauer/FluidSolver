#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>

// #include "Config.hpp"
#include "Container.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
struct FS {
  Float visc{};
  Float rho{};

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

  // Matrix<Float, NX, NY> vof{};
  // Matrix<Float, NX, NY> vof_old{};
};

template <typename Float>
struct FlowBConds {
  Float U_in;
  Float U_bot;
  Float U_top;
};

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
auto adjust_dt(const FS<Float, NX, NY>& fs, Float cfl_max, Float dt_max) -> Float {
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;
  Float CFLv_x = 0.0;
  Float CFLv_y = 0.0;

  for (size_t i = 0; i < NX; ++i) {
    for (size_t j = 0; j < NY; ++j) {
      CFLc_x = std::max(CFLc_x, (fs.U[i, j] + fs.U[i + 1, j]) / 2 / fs.dx[i]);
      CFLc_y = std::max(CFLc_y, (fs.V[i, j] + fs.V[i, j + 1]) / 2 / fs.dy[j]);
      CFLv_x = std::max(CFLv_x, 4.0 * fs.visc / (Igor::sqr(fs.dx[i]) * fs.rho));
      CFLv_y = std::max(CFLv_y, 4.0 * fs.visc / (Igor::sqr(fs.dy[j]) * fs.rho));
    }
  }

  return std::min(cfl_max / std::max({CFLc_x, CFLc_y, CFLv_x, CFLv_y}), dt_max);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
void calc_dmomdt(const FS<Float, NX, NY>& fs,
                 Matrix<Float, NX + 1, NY>& dmomUdt,
                 Matrix<Float, NX, NY + 1>& dmomVdt) {
  // TODO: Interpolate rho and visc, at the moment we assume that they are constant and it makes no
  //       difference but for two-phase flows this is not correct anymore.
  // TODO: Use the correct dx and dy in case of non-uniform grids (not planned at the moment)
  static Matrix<Float, NX, NY> FX{};
  static Matrix<Float, NX, NY> FY{};
  std::fill_n(dmomUdt.get_data(), dmomUdt.size(), 0.0);
  std::fill_n(dmomVdt.get_data(), dmomVdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  // = Calculate dmomUdt ===========================================================================
  for (size_t i = 0; i < FX.extent(0); ++i) {
    for (size_t j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
      //    = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
      FX[i, j] = -fs.rho * Igor::sqr((fs.U[i, j] + fs.U[i + 1, j]) / 2) +
                 fs.visc * (2.0 * (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] -
                            2.0 / 3.0 *
                                ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                                 (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
                 fs.p[i, j];

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FY = -rho*U*V + mu*(dUdy + dVdx)
        FY[i, j] = -fs.rho *                                                //
                       (fs.U[i, j] + fs.U[i, j + 1]) / 2 *                  //
                       (fs.V[i - 1, j + 1] + fs.V[i, j + 1]) / 2 +          //
                   fs.visc *                                                //
                       ((fs.U[i, j + 1] - fs.U[i, j]) / fs.dy[j] +          //
                        (fs.V[i, j + 1] - fs.V[i - 1, j + 1]) / fs.dx[i]);  //
      } else {
        // FY[i, j] = 0.0;
        // TODO: For debugging purposes, remove later.
        FY[i, j] = std::numeric_limits<Float>::quiet_NaN();
      }
    }
  }
  for (size_t i = 1; i < dmomUdt.extent(0) - 1; ++i) {
    for (size_t j = 1; j < dmomUdt.extent(1) - 1; ++j) {
      dmomUdt[i, j] = (FX[i, j] - FX[i - 1, j]) / fs.dx[i - 1] +  //
                      (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }

  // = Calculate dmomVdt ===========================================================================
  for (size_t i = 0; i < FX.extent(0); ++i) {
    for (size_t j = 0; j < FX.extent(1); ++j) {

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FX = -rho*U*V + mu*(dVdx + dUdy)
        FX[i, j] = -fs.rho *                                                //
                       (fs.U[i, j] + fs.U[i, j + 1]) / 2 *                  //
                       (fs.V[i - 1, j + 1] + fs.V[i, j + 1]) / 2 +          //
                   fs.visc *                                                //
                       ((fs.U[i, j + 1] - fs.U[i, j]) / fs.dy[j] +          //
                        (fs.V[i, j + 1] - fs.V[i - 1, j + 1]) / fs.dx[i]);  //
      } else {
        FX[i, j] = std::numeric_limits<Float>::quiet_NaN();
        // FX[i,j] = 0.0;
      }

      // FY = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
      //    = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
      FY[i, j] = -fs.rho * Igor::sqr((fs.V[i, j] + fs.V[i, j + 1]) / 2) +
                 fs.visc * (2.0 * (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j] -
                            2.0 / 3.0 *
                                ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                                 (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
                 fs.p[i, j];
    }
  }
  for (size_t i = 1; i < dmomVdt.extent(0) - 1; ++i) {
    for (size_t j = 1; j < dmomVdt.extent(1) - 1; ++j) {
      dmomVdt[i, j] = (FX[i + 1, j - 1] - FX[i, j - 1]) / fs.dx[i - 1] +  //
                      (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
void apply_velocity_bconds(FS<Float, NX, NY>& fs, const FlowBConds<Float>& bconds) {
  // = Boundary conditions for U-component of velocity =============================================
  for (size_t j = 0; j < fs.U.extent(1); ++j) {
    // Inflow from left
    fs.U[0, j] = bconds.U_in;

    // Outflow on right: clipped Neumann
    fs.U[fs.U.extent(0) - 1, j] = std::max(fs.U[fs.U.extent(0) - 2, j], 0.0);
  }

  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    // No-slip on bottom
    fs.U[i, 0] = (fs.U[i, 1] + 2.0 * bconds.U_bot) / 3.0;

    // No-slip on top
    fs.U[i, fs.U.extent(1) - 1] = (fs.U[i, fs.U.extent(1) - 2] + 2.0 * bconds.U_top) / 3.0;
  }

  // = Boundary conditions for V-component of velocity =============================================
  for (size_t j = 0; j < fs.V.extent(1); ++j) {
    // Inflow from left
    fs.V[0, j] = fs.V[1, j] / 3.0;  // 0.0

    // Outflow on right: clipped Neumann
    fs.V[fs.V.extent(0) - 1, j] = fs.V[fs.V.extent(0) - 2, j];
  }

  for (size_t i = 0; i < fs.V.extent(0); ++i) {
    // No-slip on bottom
    fs.V[i, 0] = 0.0;

    // No-slip on top
    fs.V[i, fs.V.extent(1) - 1] = 0.0;
  }
}

// //
// -------------------------------------------------------------------------------------------------
// void calc_dvofdt(const FS& fs, Matrix<Float, NX, NY>& dvofdt) {
//   static auto FX = make_u_staggered();
//   static auto FY = make_v_staggered();
//   std::fill_n(FX.get_data(), FX.size(), 0.0);
//   std::fill_n(FY.get_data(), FY.size(), 0.0);
//   std::fill_n(dvofdt.get_data(), dvofdt.size(), 0.0);
//
//   for (size_t i = 1; i < FX.extent(0) - 1; ++i) {
//     for (size_t j = 1; j < FX.extent(1) - 1; ++j) {
//       FX[i, j] = -(fs.vof[i, j] + fs.vof[i - 1, j]) * fs.U[i, j];
//     }
//   }
//   for (size_t i = 1; i < FY.extent(0) - 1; ++i) {
//     for (size_t j = 1; j < FY.extent(1) - 1; ++j) {
//       FY[i, j] = -(fs.vof[i, j] + fs.vof[i, j - 1]) * fs.V[i, j];
//     }
//   }
//
//   for (size_t i = 0; i < dvofdt.extent(0); ++i) {
//     for (size_t j = 0; j < dvofdt.extent(1); ++j) {
//       dvofdt[i, j] = (FX[i + 1, j] - FX[i, j]) / fs.dx[i] + (FY[i, j + 1] - FY[i, j]) / fs.dy[j];
//     }
//   }
// }
//
// //
// -------------------------------------------------------------------------------------------------
// void apply_vof_bconds(FS& fs) {
//   for (size_t j = 0; j < fs.vof.extent(1); ++j) {
//     // Neumann on left
//     fs.vof[0, j] = fs.vof[1, j];
//     // Neumann on right
//     fs.vof[fs.vof.extent(0) - 1, j] = fs.vof[fs.vof.extent(0) - 2, j];
//   }
//
//   for (size_t i = 0; i < fs.vof.extent(0); ++i) {
//     // Neumann on bottom
//     fs.vof[i, 0] = fs.vof[i, 1];
//     // Neumann on top
//     fs.vof[i, fs.vof.extent(1) - 1] = fs.vof[i, fs.vof.extent(1) - 2];
//   }
// }

#endif  // FLUID_SOLVER_FS_HPP_
