#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/MdArray.hpp>

#include "Config.hpp"

// -------------------------------------------------------------------------------------------------
struct FS {
  Igor::MdArray<Float, NX_P1_EXTENT> x;
  Igor::MdArray<Float, NX_EXTENT> xm;
  Igor::MdArray<Float, NX_EXTENT> dx;

  Igor::MdArray<Float, NY_P1_EXTENT> y;
  Igor::MdArray<Float, NY_EXTENT> ym;
  Igor::MdArray<Float, NY_EXTENT> dy;

  Igor::MdArray<Float, CENTERED_EXTENT> rho;
  Igor::MdArray<Float, CENTERED_EXTENT> rho_old;

  Igor::MdArray<Float, U_STAGGERED_EXTENT> U;
  Igor::MdArray<Float, U_STAGGERED_EXTENT> U_old;

  Igor::MdArray<Float, V_STAGGERED_EXTENT> V;
  Igor::MdArray<Float, V_STAGGERED_EXTENT> V_old;

  Igor::MdArray<Float, CENTERED_EXTENT> p;

  Igor::MdArray<Float, CENTERED_EXTENT> visc;
};

// -------------------------------------------------------------------------------------------------
auto adjust_dt(const FS& fs, Float dt_old) -> Float {
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;

  for (size_t i = 0; i < NX; ++i) {
    for (size_t j = 0; j < NY; ++j) {
      CFLc_x = std::max(CFLc_x, (fs.U[i, j] + fs.U[i + 1, j]) / 2 / fs.dx[i]);
      CFLc_y = std::max(CFLc_y, (fs.V[i, j] + fs.V[i, j + 1]) / 2 / fs.dy[j]);
    }
  }
  CFLc_x *= dt_old;
  CFLc_y *= dt_old;

  return std::min(dt_old * CFL_MAX / std::max(CFLc_x, CFLc_y), DT_MAX);
}

// -------------------------------------------------------------------------------------------------
void calc_dmomdt(const FS& fs,
                 Igor::MdArray<Float, U_STAGGERED_EXTENT>& dmomUdt,
                 Igor::MdArray<Float, V_STAGGERED_EXTENT>& dmomVdt) {
  static auto FX = make_centered();
  static auto FY = make_centered();
  std::fill_n(dmomUdt.get_data(), dmomUdt.size(), 0.0);
  std::fill_n(dmomVdt.get_data(), dmomVdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  // - Calculate dmomUdt ---------------------------------------------------------------------------
  for (size_t i = 0; i < FX.extent(0); ++i) {
    for (size_t j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
      FX[i, j] =
          -fs.rho[i, j] * (fs.U[i, j] + fs.U[i + 1, j]) / 2 * (fs.U[i, j] + fs.U[i + 1, j]) / 2 +
          fs.visc[i, j] *
              ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] + (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] -
               2.0 / 3.0 *
                   ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                    (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -

          fs.p[i, j];
      // FY = -rho*U*V + mu*(dUdx + dVdy)
      FY[i, j] =
          -fs.rho[i, j] * (fs.U[i, j] + fs.U[i + 1, j]) / 2 * (fs.V[i, j] + fs.V[i, j + 1]) / 2 +
          fs.visc[i, j] *
              ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] + (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j]);
    }
  }
  for (size_t i = 1; i < dmomUdt.extent(0) - 1; ++i) {
    for (size_t j = 1; j < dmomUdt.extent(1) - 1; ++j) {
      dmomUdt[i, j] =
          (FX[i, j] - FX[i - 1, j]) / fs.dx[i - 1] + (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }

  // - Calculate dmomVdt ---------------------------------------------------------------------------
  for (size_t i = 0; i < FX.extent(0); ++i) {
    for (size_t j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*V + mu*(dUdx + dVdy)
      FX[i, j] =
          -fs.rho[i, j] * (fs.U[i, j] + fs.U[i + 1, j]) / 2 * (fs.V[i, j] + fs.V[i, j + 1]) / 2 +
          fs.visc[i, j] *
              ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] + (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j]);
      // FY = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
      FY[i, j] =
          -fs.rho[i, j] * (fs.V[i, j] + fs.V[i, j + 1]) / 2 * (fs.V[i, j] + fs.V[i, j + 1]) / 2 +
          fs.visc[i, j] *
              ((fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j] + (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j] -
               2.0 / 3.0 *
                   ((fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +
                    (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j])) -
          fs.p[i, j];
    }
  }
  for (size_t i = 1; i < dmomUdt.extent(0) - 1; ++i) {
    for (size_t j = 1; j < dmomUdt.extent(1) - 1; ++j) {
      dmomVdt[i, j] =
          (FX[i, j] - FX[i - 1, j]) / fs.dx[i - 1] + (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }
}

// -------------------------------------------------------------------------------------------------
void apply_bconds(FS& fs) {
  for (size_t i = 0; i < fs.U.extent(0); ++i) {
    // No-slip on bottom
    fs.U[i, 0] = 0.0;
    fs.V[i, 0] = 0.0;

    // No-slip on top
    fs.U[i, fs.U.extent(1) - 1] = 0.0;
    fs.V[i, fs.V.extent(1) - 1] = 0.0;
  }

  for (size_t j = 0; j < fs.U.extent(1); ++j) {
    // Inflow from left
    fs.U[0, j] = U_IN;
    fs.V[0, j] = 0.0;

    // Outflow on right
    fs.U[fs.U.extent(0) - 1, j] = fs.U[fs.U.extent(0) - 2, j];
    fs.V[fs.V.extent(0) - 1, j] = fs.V[fs.V.extent(0) - 2, j];
  }
}

#endif  // FLUID_SOLVER_FS_HPP_
