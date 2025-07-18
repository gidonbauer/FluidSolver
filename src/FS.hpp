#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>

#include "Container.hpp"
#include "IR.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
struct State {
  Matrix<Float, NX + 1, NY> rho_u_stag{};
  Matrix<Float, NX, NY + 1> rho_v_stag{};

  Matrix<Float, NX + 1, NY> U{};
  Matrix<Float, NX, NY + 1> V{};
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
struct FS {
  Float visc_gas{};
  Float visc_liquid{};
  Float rho_gas{};
  Float rho_liquid{};

  Vector<Float, NX + 1> x{};
  Vector<Float, NX> xm{};
  Vector<Float, NX> dx{};

  Vector<Float, NY + 1> y{};
  Vector<Float, NY> ym{};
  Vector<Float, NY> dy{};

  Matrix<Float, NX, NY> visc{};

  Matrix<Float, NX, NY> p{};

  State<Float, NX, NY> old{};
  State<Float, NX, NY> curr{};
};

template <typename Float, Index NX, Index NY>
constexpr void save_old_velocity(const State<Float, NX, NY>& curr,
                                 State<Float, NX, NY>& old) noexcept {
  std::copy_n(curr.U.get_data(), curr.U.size(), old.U.get_data());
  std::copy_n(curr.V.get_data(), curr.V.size(), old.V.get_data());
}

template <typename Float, Index NX, Index NY>
constexpr void save_old_density(const State<Float, NX, NY>& curr,
                                State<Float, NX, NY>& old) noexcept {
  std::copy_n(curr.rho_u_stag.get_data(), curr.rho_u_stag.size(), old.rho_u_stag.get_data());
  std::copy_n(curr.rho_v_stag.get_data(), curr.rho_v_stag.size(), old.rho_v_stag.get_data());
}

template <typename Float, Index NX, Index NY>
constexpr void save_old_state(const State<Float, NX, NY>& curr,
                              State<Float, NX, NY>& old) noexcept {
  save_old_density(curr, old);
  save_old_velocity(curr, old);
}

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
  // TODO: Is this correct for the two-phase case with high density differences?
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;
  Float CFLv_x = 0.0;
  Float CFLv_y = 0.0;

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      CFLc_x         = std::max(CFLc_x, (fs.curr.U[i, j] + fs.curr.U[i + 1, j]) / 2 / fs.dx[i]);
      CFLc_y         = std::max(CFLc_y, (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2 / fs.dy[j]);

      const auto rho = (fs.curr.rho_u_stag[i, j] + fs.curr.rho_u_stag[i + 1, j] +
                        fs.curr.rho_v_stag[i, j] + fs.curr.rho_v_stag[i, j + 1]) /
                       4.0;
      CFLv_x = std::max(CFLv_x, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dx[i]) * rho));
      CFLv_y = std::max(CFLv_y, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dy[j]) * rho));
    }
  }

  return std::min(cfl_max / std::max({CFLc_x, CFLc_y, CFLv_x, CFLv_y}), dt_max);
}

// -------------------------------------------------------------------------------------------------
// Hybrid interpolation scheme for high density jumps
template <typename Float>
constexpr auto hybrid_interp(Float rho_eps,
                             Float interp_rho_minus,
                             Float interp_rho_plus,
                             Float interp_velo_minus,
                             Float interp_velo_plus,
                             Float transp_velo_minus,
                             Float transp_velo_plus) -> std::array<Float, 2> {
  const auto use_upwind = std::abs(interp_rho_plus - interp_rho_minus) > rho_eps;

  if (!use_upwind) {
    return {
        (interp_rho_plus + interp_rho_minus) / 2.0,
        (interp_velo_plus + interp_velo_minus) / 2.0,
    };
  }
  if (transp_velo_plus + transp_velo_minus >= 0.0) { return {interp_rho_minus, interp_velo_minus}; }
  return {interp_rho_plus, interp_velo_plus};
};

template <typename Float, Index NX, Index NY>
constexpr auto calc_rho_eps(const FS<Float, NX, NY>& fs) noexcept -> Float {
  return 1e-3 * std::min(fs.rho_gas, fs.rho_liquid);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_dmomdt(const FS<Float, NX, NY>& fs,
                 Matrix<Float, NX + 1, NY>& dmomUdt,
                 Matrix<Float, NX, NY + 1>& dmomVdt) {
  // TODO: Use the correct dx and dy in case of non-uniform grids (not planned at the moment)
  static Matrix<Float, NX, NY> FX{};
  static Matrix<Float, NX, NY> FY{};
  std::fill_n(dmomUdt.get_data(), dmomUdt.size(), 0.0);
  std::fill_n(dmomVdt.get_data(), dmomVdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  const auto rho_eps = calc_rho_eps(fs);

  // = Calculate dmomUdt ===========================================================================
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {
      // FX = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
      //    = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
      //    = -rho*U^2 + 2*mu*dUdx - p

      // = On center mesh ========================
      {
        const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                              fs.old.rho_u_stag[i, j],
                                                              fs.old.rho_u_stag[i + 1, j],
                                                              fs.curr.U[i, j],
                                                              fs.curr.U[i + 1, j],
                                                              fs.curr.U[i, j],
                                                              fs.curr.U[i + 1, j]);
        const auto U_i                        = ((fs.curr.U[i + 1, j] + fs.curr.U[i, j]) / 2);
        const auto dUdx                       = (fs.curr.U[i + 1, j] - fs.curr.U[i, j]) / fs.dx[i];

        FX[i, j] = -rho_i_hybrid * U_i_hybrid * U_i + 2.0 * fs.visc[i, j] * dUdx - fs.p[i, j];
      }

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FY = -rho*U*V + mu*(dUdy + dVdx)

        // = On corner mesh ======================
        const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                              fs.old.rho_u_stag[i, j],
                                                              fs.old.rho_u_stag[i, j + 1],
                                                              fs.curr.U[i, j],
                                                              fs.curr.U[i, j + 1],
                                                              fs.curr.V[i - 1, j + 1],
                                                              fs.curr.V[i, j + 1]);
        const auto V_i                        = (fs.curr.V[i - 1, j + 1] + fs.curr.V[i, j + 1]) / 2;

        const auto visc_corner =
            (fs.visc[i - 1, j] + fs.visc[i, j] + fs.visc[i - 1, j + 1] + fs.visc[i, j + 1]) / 4.0;
        const auto dUdy = (fs.curr.U[i, j + 1] - fs.curr.U[i, j]) / fs.dy[j];
        const auto dVdx = (fs.curr.V[i, j + 1] - fs.curr.V[i - 1, j + 1]) / fs.dx[i];

        FY[i, j]        = -rho_i_hybrid * U_i_hybrid * V_i + visc_corner * (dUdy + dVdx);
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

        // = On corner mesh ======================
        const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                              fs.old.rho_v_stag[i - 1, j + 1],
                                                              fs.old.rho_v_stag[i, j + 1],
                                                              fs.curr.V[i - 1, j + 1],
                                                              fs.curr.V[i, j + 1],
                                                              fs.curr.U[i, j],
                                                              fs.curr.U[i, j + 1]);
        const auto U_i                        = (fs.curr.U[i, j] + fs.curr.U[i, j + 1]) / 2;

        const auto visc_corner =
            (fs.visc[i - 1, j] + fs.visc[i, j] + fs.visc[i - 1, j + 1] + fs.visc[i, j + 1]) / 4.0;
        const auto dUdy = (fs.curr.U[i, j + 1] - fs.curr.U[i, j]) / fs.dy[j];
        const auto dVdx = (fs.curr.V[i, j + 1] - fs.curr.V[i - 1, j + 1]) / fs.dx[i];

        FX[i, j]        = -rho_i_hybrid * U_i * V_i_hybrid + visc_corner * (dUdy + dVdx);
      } else {
        FX[i, j] = std::numeric_limits<Float>::quiet_NaN();
        // FX[i,j] = 0.0;
      }

      // FY = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
      //    = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
      //    = -rho*V^2 + 2*mu*dVdy - p

      // = On center mesh ========================
      {
        const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                              fs.old.rho_v_stag[i, j],
                                                              fs.old.rho_v_stag[i, j + 1],
                                                              fs.curr.V[i, j],
                                                              fs.curr.V[i, j + 1],
                                                              fs.curr.V[i, j],
                                                              fs.curr.V[i, j + 1]);
        const auto V_i                        = (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2;

        const auto dVdy                       = (fs.curr.V[i, j + 1] - fs.curr.V[i, j]) / fs.dy[j];

        FY[i, j] = -rho_i_hybrid * V_i_hybrid * V_i + 2.0 * fs.visc[i, j] * dVdy - fs.p[i, j];
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
void calc_drhodt(const FS<Float, NX, NY>& fs,
                 Matrix<Float, NX + 1, NY>& drho_u_stagdt,
                 Matrix<Float, NX, NY + 1>& drho_v_stagdt) {
  static Matrix<Float, NX, NY> FX{};
  static Matrix<Float, NX, NY> FY{};
  std::fill_n(drho_u_stagdt.get_data(), drho_u_stagdt.size(), 0.0);
  std::fill_n(drho_v_stagdt.get_data(), drho_v_stagdt.size(), 0.0);
  std::fill_n(FX.get_data(), FX.size(), 0.0);
  std::fill_n(FY.get_data(), FY.size(), 0.0);

  const auto rho_eps = calc_rho_eps(fs);

  // = Calculate drhodt for U-staggered density ====================================================
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {
      // FX = -rho * U
      {
        const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                     fs.old.rho_u_stag[i, j],
                                                     fs.old.rho_u_stag[i + 1, j],
                                                     0.0,
                                                     0.0,
                                                     fs.curr.U[i, j],
                                                     fs.curr.U[i + 1, j]);
        const auto U_i               = (fs.curr.U[i, j] + fs.curr.U[i + 1, j]) / 2.0;
        FX[i, j]                     = -rho_i_hybrid * U_i;
      }

      if (i > 0 && j < FX.extent(1) - 1) {
        // FY = -rho * V
        const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                     fs.old.rho_u_stag[i, j],
                                                     fs.old.rho_u_stag[i, j + 1],
                                                     0.0,
                                                     0.0,
                                                     fs.curr.V[i - 1, j + 1],
                                                     fs.curr.V[i, j + 1]);
        const auto V_i               = (fs.curr.V[i - 1, j + 1] + fs.curr.V[i, j + 1]) / 2;
        FY[i, j]                     = -rho_i_hybrid * V_i;
      } else {
        // TODO: For debugging purposes, remove later.
        FY[i, j] = std::numeric_limits<Float>::quiet_NaN();
      }
    }
  }
  for (Index i = 1; i < drho_u_stagdt.extent(0) - 1; ++i) {
    for (Index j = 1; j < drho_u_stagdt.extent(1) - 1; ++j) {
      drho_u_stagdt[i, j] = (FX[i, j] - FX[i - 1, j]) / fs.dx[i - 1] +  //
                            (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }

  // = Calculate drhodt for V-staggered density ====================================================
  for (Index i = 0; i < FX.extent(0); ++i) {
    for (Index j = 0; j < FX.extent(1); ++j) {

      // Prevent accessing U and V out of bounds
      if (i > 0 && j < FX.extent(1) - 1) {
        // FX = -rho*U

        // = On corner mesh ======================
        const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                     fs.old.rho_v_stag[i - 1, j + 1],
                                                     fs.old.rho_v_stag[i, j + 1],
                                                     0.0,
                                                     0.0,
                                                     fs.curr.U[i, j],
                                                     fs.curr.U[i, j + 1]);
        const auto U_i               = (fs.curr.U[i, j] + fs.curr.U[i, j + 1]) / 2.0;
        FX[i, j]                     = -rho_i_hybrid * U_i;
      } else {
        FX[i, j] = std::numeric_limits<Float>::quiet_NaN();
      }

      // = On center mesh ========================
      {
        // FY = -rho*V
        const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                     fs.old.rho_v_stag[i, j],
                                                     fs.old.rho_v_stag[i, j + 1],
                                                     0.0,
                                                     0.0,
                                                     fs.curr.V[i, j],
                                                     fs.curr.V[i, j + 1]);
        const auto V_i               = (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2.0;
        FY[i, j]                     = -rho_i_hybrid * V_i;
      }
    }
  }
  for (Index i = 1; i < drho_v_stagdt.extent(0) - 1; ++i) {
    for (Index j = 1; j < drho_v_stagdt.extent(1) - 1; ++j) {
      drho_v_stagdt[i, j] = (FX[i + 1, j - 1] - FX[i, j - 1]) / fs.dx[i - 1] +  //
                            (FY[i, j] - FY[i, j - 1]) / fs.dy[j - 1];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void apply_velocity_bconds(FS<Float, NX, NY>& fs, const FlowBConds<Float>& bconds) {
  // = Boundary conditions for U-component of velocity =============================================
  for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET: fs.curr.U[0, j] = bconds.U[LEFT]; break;
      case BCond::NEUMANN:   fs.curr.U[0, j] = fs.curr.U[1, j]; break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET: fs.curr.U[fs.curr.U.extent(0) - 1, j] = bconds.U[RIGHT]; break;
      case BCond::NEUMANN:
        fs.curr.U[fs.curr.U.extent(0) - 1, j] = fs.curr.U[fs.curr.U.extent(0) - 2, j];
        break;
    }
  }

  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET:
        fs.curr.U[i, 0] = (fs.curr.U[i, 1] + 2.0 * bconds.U[BOTTOM]) / 3.0;
        break;
      case BCond::NEUMANN: fs.curr.U[i, 0] = fs.curr.U[i, 1]; break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET:
        fs.curr.U[i, fs.curr.U.extent(1) - 1] =
            (fs.curr.U[i, fs.curr.U.extent(1) - 2] + 2.0 * bconds.U[TOP]) / 3.0;
        break;
      case BCond::NEUMANN:
        fs.curr.U[i, fs.curr.U.extent(1) - 1] = fs.curr.U[i, fs.curr.U.extent(1) - 2];
        break;
    }
  }

  // = Boundary conditions for V-component of velocity =============================================
  for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET:
        fs.curr.V[0, j] = (fs.curr.V[1, j] + 2.0 * bconds.V[LEFT]) / 3.0;
        break;
      case BCond::NEUMANN: fs.curr.V[0, j] = fs.curr.V[1, j]; break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET:
        fs.curr.V[fs.curr.V.extent(0) - 1, j] =
            (fs.curr.V[fs.curr.V.extent(0) - 2, j] + 2.0 * bconds.V[RIGHT]) / 3.0;
        break;
      case BCond::NEUMANN:
        fs.curr.V[fs.curr.V.extent(0) - 1, j] = fs.curr.V[fs.curr.V.extent(0) - 2, j];
        break;
    }
  }

  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET: fs.curr.V[i, 0] = bconds.V[BOTTOM]; break;
      case BCond::NEUMANN:   fs.curr.V[i, 0] = fs.curr.V[i, 1]; break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET: fs.curr.V[i, fs.curr.V.extent(1) - 1] = bconds.V[TOP]; break;
      case BCond::NEUMANN:
        fs.curr.V[i, fs.curr.V.extent(1) - 1] = fs.curr.V[i, fs.curr.V.extent(1) - 2];
        break;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr void apply_neumann_bconds(Matrix<Float, NX, NY>& field) noexcept {
  for (Index j = 0; j < field.extent(1); ++j) {
    // LEFT
    field[0, j] = field[1, j];
    // RIGHT
    field[field.extent(0) - 1, j] = field[field.extent(0) - 2, j];
  }

  for (Index i = 0; i < field.extent(0); ++i) {
    // BOTTOM
    field[i, 0] = field[i, 1];
    // TOP
    field[i, field.extent(1) - 1] = field[i, field.extent(1) - 2];
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
constexpr void calc_rho_and_visc(FS<Float, NX, NY>& fs) noexcept {
  IGOR_ASSERT(std::abs(fs.rho_gas - fs.rho_liquid) < 1e-12,
              "Expected constant density but rho_gas = {:.6e} and rho_liquid = {:.6e}",
              fs.rho_gas,
              fs.rho_liquid);
  IGOR_ASSERT(std::abs(fs.visc_gas - fs.visc_liquid) < 1e-12,
              "Expected constant viscosity but visc_gas = {:.6e} and visc_liquid = {:.6e}",
              fs.visc_gas,
              fs.visc_liquid);

  std::fill_n(fs.old.rho_u_stag.get_data(), fs.old.rho_u_stag.size(), fs.rho_gas);
  std::fill_n(fs.old.rho_v_stag.get_data(), fs.old.rho_v_stag.size(), fs.rho_gas);
  std::fill_n(fs.curr.rho_u_stag.get_data(), fs.curr.rho_u_stag.size(), fs.rho_gas);
  std::fill_n(fs.curr.rho_v_stag.get_data(), fs.curr.rho_v_stag.size(), fs.rho_gas);
  std::fill_n(fs.visc.get_data(), fs.visc.size(), fs.visc_gas);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr void calc_rho_and_visc(const InterfaceReconstruction<NX, NY>& ir,
                                 const Matrix<Float, NX, NY>& vof,
                                 FS<Float, NX, NY>& fs) noexcept {
#if 0
  (void)ir;
  // = Density on U-staggered mesh =================================================================
  for (Index i = 1; i < fs.curr.rho_u_stag.extent(0) - 1; ++i) {
    for (Index j = 0; j < fs.curr.rho_u_stag.extent(1); ++j) {
      const auto rho_minus     = vof[i - 1, j] * fs.rho_liquid + (1.0 - vof[i - 1, j]) * fs.rho_gas;
      const auto rho_plus      = vof[i, j] * fs.rho_liquid + (1.0 - vof[i, j]) * fs.rho_gas;
      fs.curr.rho_u_stag[i, j] = (rho_minus + rho_plus) / 2.0;
    }
  }
  for (Index j = 0; j < fs.curr.rho_u_stag.extent(1); ++j) {
    fs.curr.rho_u_stag[0, j]  = fs.curr.rho_u_stag[1, j];
    fs.curr.rho_u_stag[NX, j] = fs.curr.rho_u_stag[NX - 1, j];
  }

  // = Density on V-staggered mesh =================================================================
  for (Index i = 0; i < fs.curr.rho_v_stag.extent(0); ++i) {
    for (Index j = 1; j < fs.curr.rho_v_stag.extent(1) - 1; ++j) {
      const auto rho_minus     = vof[i, j - 1] * fs.rho_liquid + (1.0 - vof[i, j - 1]) * fs.rho_gas;
      const auto rho_plus      = vof[i, j] * fs.rho_liquid + (1.0 - vof[i, j]) * fs.rho_gas;
      fs.curr.rho_v_stag[i, j] = (rho_minus + rho_plus) / 2.0;
    }
  }
  for (Index i = 0; i < fs.curr.rho_v_stag.extent(0); ++i) {
    fs.curr.rho_v_stag[i, 0]  = fs.curr.rho_v_stag[i, 1];
    fs.curr.rho_v_stag[i, NY] = fs.curr.rho_v_stag[i, NY - 1];
  }
#else
  auto get_dist = [](const IRL::PlanarSeparator& interface, const IRL::Pt& pt) -> Float {
    IGOR_ASSERT(interface.getNumberOfPlanes() == 1,
                "Expected one plane but got {}",
                interface.getNumberOfPlanes());
    const auto& plane = interface[0];
    return plane.signedDistanceToPoint(pt);
  };

  // = Density on U-staggered mesh =================================================================
  for (Index i = 1; i < fs.curr.rho_u_stag.extent(0) - 1; ++i) {
    for (Index j = 1; j < fs.curr.rho_u_stag.extent(1) - 1; ++j) {
      const auto vof_i = (vof[i - 1, j] + vof[i, j]) / 2.0;
      if (vof_i >= VOF_HIGH) {
        fs.curr.rho_u_stag[i, j] = fs.rho_liquid;
      } else if (vof_i <= VOF_LOW) {
        fs.curr.rho_u_stag[i, j] = fs.rho_gas;
      } else {
        const auto minus_has_interface = has_interface(vof, i - 1, j);
        const auto plus_has_interface  = has_interface(vof, i, j);
        const IRL::Pt pt{fs.x[i], fs.ym[j], 0.0};

        if (minus_has_interface && plus_has_interface) {
          const auto dist1 = get_dist(ir.interface[i - 1, j], pt);
          const auto dist2 = get_dist(ir.interface[i, j], pt);
          if (dist1 > 0.0 && dist2 > 0.0) {
            fs.curr.rho_u_stag[i, j] = fs.rho_gas;
          } else if (dist1 <= 0.0 && dist2 <= 0.0) {
            fs.curr.rho_u_stag[i, j] = fs.rho_liquid;
          } else {
            const auto rho_minus =
                vof[i - 1, j] * fs.rho_liquid + (1.0 - vof[i - 1, j]) * fs.rho_gas;
            const auto rho_plus      = vof[i, j] * fs.rho_liquid + (1.0 - vof[i, j]) * fs.rho_gas;
            fs.curr.rho_u_stag[i, j] = (rho_minus + rho_plus) / 2.0;
          }
        } else if (minus_has_interface) {
          const auto dist = get_dist(ir.interface[i - 1, j], pt);
          if (dist > 0.0) {
            fs.curr.rho_u_stag[i, j] = fs.rho_gas;
          } else {
            fs.curr.rho_u_stag[i, j] = fs.rho_liquid;
          }
        } else if (plus_has_interface) {
          const auto dist = get_dist(ir.interface[i, j], pt);
          if (dist > 0.0) {
            fs.curr.rho_u_stag[i, j] = fs.rho_gas;
          } else {
            fs.curr.rho_u_stag[i, j] = fs.rho_liquid;
          }
        } else {
          Igor::Panic("Unreachable: One of the neighboring cells must have an interface.");
        }
      }
    }
  }
  apply_neumann_bconds(fs.curr.rho_u_stag);

  // = Density on V-staggered mesh =================================================================
  for (Index i = 1; i < fs.curr.rho_v_stag.extent(0) - 1; ++i) {
    for (Index j = 1; j < fs.curr.rho_v_stag.extent(1) - 1; ++j) {
      const auto vof_i = (vof[i, j - 1] + vof[i, j]) / 2.0;
      if (vof_i >= VOF_HIGH) {
        fs.curr.rho_v_stag[i, j] = fs.rho_liquid;
      } else if (vof_i <= VOF_LOW) {
        fs.curr.rho_v_stag[i, j] = fs.rho_gas;
      } else {
        const auto minus_has_interface = has_interface(vof, i, j - 1);
        const auto plus_has_interface  = has_interface(vof, i, j);
        const IRL::Pt pt{fs.xm[i], fs.y[j], 0.0};

        if (minus_has_interface && plus_has_interface) {
          const auto dist1 = get_dist(ir.interface[i, j - 1], pt);
          const auto dist2 = get_dist(ir.interface[i, j], pt);
          if (dist1 > 0.0 && dist2 > 0.0) {
            fs.curr.rho_v_stag[i, j] = fs.rho_gas;
          } else if (dist1 <= 0.0 && dist2 <= 0.0) {
            fs.curr.rho_v_stag[i, j] = fs.rho_liquid;
          } else {
            const auto rho_minus =
                vof[i, j - 1] * fs.rho_liquid + (1.0 - vof[i, j - 1]) * fs.rho_gas;
            const auto rho_plus      = vof[i, j] * fs.rho_liquid + (1.0 - vof[i, j]) * fs.rho_gas;
            fs.curr.rho_v_stag[i, j] = (rho_minus + rho_plus) / 2.0;
          }
        } else if (minus_has_interface) {
          const auto dist = get_dist(ir.interface[i, j - 1], pt);
          if (dist > 0.0) {
            fs.curr.rho_v_stag[i, j] = fs.rho_gas;
          } else {
            fs.curr.rho_v_stag[i, j] = fs.rho_liquid;
          }
        } else if (plus_has_interface) {
          const auto dist = get_dist(ir.interface[i, j], pt);
          if (dist > 0.0) {
            fs.curr.rho_v_stag[i, j] = fs.rho_gas;
          } else {
            fs.curr.rho_v_stag[i, j] = fs.rho_liquid;
          }
        } else {
          Igor::Panic("Unreachable: One of the neighboring cells must have an interface.");
        }
      }
    }
  }
  apply_neumann_bconds(fs.curr.rho_v_stag);
#endif

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      fs.visc[i, j] = vof[i, j] * fs.visc_liquid + (1.0 - vof[i, j]) * fs.visc_gas;
    }
  }
}

#endif  // FLUID_SOLVER_FS_HPP_
