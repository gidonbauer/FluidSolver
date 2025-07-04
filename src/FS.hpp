#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>

// #include "Config.hpp"
#include "Container.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
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
      CFLv_x = std::max(CFLv_x, 4.0 * fs.visc / (Igor::sqr(fs.dx[i]) * fs.rho));
      CFLv_y = std::max(CFLv_y, 4.0 * fs.visc / (Igor::sqr(fs.dy[j]) * fs.rho));
    }
  }

  return std::min(cfl_max / std::max({CFLc_x, CFLc_y, CFLv_x, CFLv_y}), dt_max);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
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
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {
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

#endif  // FLUID_SOLVER_FS_HPP_
