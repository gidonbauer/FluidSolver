#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>
#include <Igor/MdArray.hpp>

#include "Config.hpp"

// -------------------------------------------------------------------------------------------------
struct FS {
  Igor::MdArray<Float, NX_P1_EXTENT> x = make_nx_p1();
  Igor::MdArray<Float, NX_EXTENT> xm   = make_nx();
  Igor::MdArray<Float, NX_EXTENT> dx   = make_nx();

  Igor::MdArray<Float, NY_P1_EXTENT> y = make_ny_p1();
  Igor::MdArray<Float, NY_EXTENT> ym   = make_ny();
  Igor::MdArray<Float, NY_EXTENT> dy   = make_ny();

  Igor::MdArray<Float, U_STAGGERED_EXTENT> U     = make_u_staggered();
  Igor::MdArray<Float, U_STAGGERED_EXTENT> U_old = make_u_staggered();

  Igor::MdArray<Float, V_STAGGERED_EXTENT> V     = make_v_staggered();
  Igor::MdArray<Float, V_STAGGERED_EXTENT> V_old = make_v_staggered();

  Igor::MdArray<Float, CENTERED_EXTENT> p = make_centered();

  Igor::MdArray<Float, CENTERED_EXTENT> vof     = make_centered();
  Igor::MdArray<Float, CENTERED_EXTENT> vof_old = make_centered();
};

// -------------------------------------------------------------------------------------------------
auto adjust_dt(const FS& fs) -> Float {
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;
  Float CFLv_x = 0.0;
  Float CFLv_y = 0.0;

  for (size_t i = 0; i < NX; ++i) {
    for (size_t j = 0; j < NY; ++j) {
      CFLc_x = std::max(CFLc_x, (fs.U[i, j] + fs.U[i + 1, j]) / 2 / fs.dx[i]);
      CFLc_y = std::max(CFLc_y, (fs.V[i, j] + fs.V[i, j + 1]) / 2 / fs.dy[j]);
      CFLv_x = std::max(CFLv_x, 4.0 * VISC / (Igor::sqr(fs.dx[i]) * RHO));
      CFLv_y = std::max(CFLv_y, 4.0 * VISC / (Igor::sqr(fs.dy[j]) * RHO));
    }
  }

  return std::min(CFL_MAX / std::max({CFLc_x, CFLc_y, CFLv_x, CFLv_y}), DT_MAX);
}

// -------------------------------------------------------------------------------------------------
void calc_dmomdt(const FS& fs,
                 Igor::MdArray<Float, U_STAGGERED_EXTENT>& dmomUdt,
                 Igor::MdArray<Float, V_STAGGERED_EXTENT>& dmomVdt) {
  // TODO: Interpolate rho and visc, at the moment we assume that they are constant and it makes no
  //       difference but for two-phase flows this is not correct anymore.
  // TODO: Use the correct dx and dy in case of non-uniform grids (not planned at the moment)
  static auto FX = make_centered();
  static auto FY = make_centered();
  std::fill_n(dmomUdt.get_data(), dmomUdt.size(), 0.0);
  std::fill_n(dmomVdt.get_data(), dmomVdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  // = Calculate dmomUdt ===========================================================================
  for (size_t i = 0; i < FX.extent(0); ++i) {
    for (size_t j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
      //    = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
      FX[i, j] = -RHO * Igor::sqr((fs.U[i, j] + fs.U[i + 1, j]) / 2) +
                 VISC * (2.0 * (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] -
                         2.0 / 3.0 *
                             ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                              (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
                 fs.p[i, j];

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FY = -rho*U*V + mu*(dUdy + dVdx)
        FY[i, j] = -RHO *                                                   //
                       (fs.U[i, j] + fs.U[i, j + 1]) / 2 *                  //
                       (fs.V[i - 1, j + 1] + fs.V[i, j + 1]) / 2 +          //
                   VISC *                                                   //
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
        FX[i, j] = -RHO *                                                   //
                       (fs.U[i, j] + fs.U[i, j + 1]) / 2 *                  //
                       (fs.V[i - 1, j + 1] + fs.V[i, j + 1]) / 2 +          //
                   VISC *                                                   //
                       ((fs.U[i, j + 1] - fs.U[i, j]) / fs.dy[j] +          //
                        (fs.V[i, j + 1] - fs.V[i - 1, j + 1]) / fs.dx[i]);  //
      } else {
        FX[i, j] = std::numeric_limits<Float>::quiet_NaN();
        // FX[i,j] = 0.0;
      }

      // FY = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
      //    = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
      FY[i, j] = -RHO * Igor::sqr((fs.V[i, j] + fs.V[i, j + 1]) / 2) +
                 VISC * (2.0 * (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j] -
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
void apply_velocity_bconds(FS& fs) {
  // = Boundary conditions for U-component of velocity =============================================
  for (size_t j = 0; j < fs.U.extent(1); ++j) {
    // Inflow from left
    fs.U[0, j] = U_IN;

    // Outflow on right: clipped Neumann
    fs.U[fs.U.extent(0) - 1, j] = std::max(fs.U[fs.U.extent(0) - 2, j], 0.0);
  }

  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    // No-slip on bottom
    fs.U[i, 0] = (fs.U[i, 1] + 2.0 * U_BOT) / 3.0;

    // No-slip on top
    fs.U[i, fs.U.extent(1) - 1] = (fs.U[i, fs.U.extent(1) - 2] + 2.0 * U_TOP) / 3.0;
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

// -------------------------------------------------------------------------------------------------
void calc_dvofdt(const FS& fs, Igor::MdArray<Float, CENTERED_EXTENT>& dvofdt) {
  static auto FX = make_u_staggered();
  static auto FY = make_v_staggered();
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);
  std::fill_n(dvofdt.get_data(), dvofdt.size(), 0.0);

  for (size_t i = 1; i < FX.extent(0) - 1; ++i) {
    for (size_t j = 1; j < FX.extent(1) - 1; ++j) {
      FX[i, j] = -(fs.vof[i, j] + fs.vof[i - 1, j]) * fs.U[i, j];
    }
  }
  for (size_t i = 1; i < FY.extent(0) - 1; ++i) {
    for (size_t j = 1; j < FY.extent(1) - 1; ++j) {
      FY[i, j] = -(fs.vof[i, j] + fs.vof[i, j - 1]) * fs.V[i, j];
    }
  }

  for (size_t i = 0; i < dvofdt.extent(0); ++i) {
    for (size_t j = 0; j < dvofdt.extent(1); ++j) {
      dvofdt[i, j] = (FX[i + 1, j] - FX[i, j]) / fs.dx[i] + (FY[i, j + 1] - FY[i, j]) / fs.dy[j];
    }
  }
}

// -------------------------------------------------------------------------------------------------
void apply_vof_bconds(FS& fs) {
  for (size_t j = 0; j < fs.vof.extent(1); ++j) {
    // Neumann on left
    fs.vof[0, j] = fs.vof[1, j];
    // Neumann on right
    fs.vof[fs.vof.extent(0) - 1, j] = fs.vof[fs.vof.extent(0) - 2, j];
  }

  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    // Neumann on bottom
    fs.vof[i, 0] = fs.vof[i, 1];
    // Neumann on top
    fs.vof[i, fs.vof.extent(1) - 1] = fs.vof[i, fs.vof.extent(1) - 2];
  }
}

#endif  // FLUID_SOLVER_FS_HPP_
