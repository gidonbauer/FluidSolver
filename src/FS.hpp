#ifndef FLUID_SOLVER_FS_HPP_
#define FLUID_SOLVER_FS_HPP_

#include <algorithm>

#include <Igor/Math.hpp>

#include "BoundaryConditions.hpp"
#include "Container.hpp"
#include "ForEach.hpp"
#include "IR.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
struct State {
  Matrix<Float, NX + 1, NY, NGHOST> rho_u_stag{};
  Matrix<Float, NX, NY + 1, NGHOST> rho_v_stag{};

  Matrix<Float, NX + 1, NY, NGHOST> U{};
  Matrix<Float, NX, NY + 1, NGHOST> V{};
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
requires(NGHOST > 0)
struct FS {
  Float visc_gas{};
  Float visc_liquid{};
  Float rho_gas{};
  Float rho_liquid{};

  Vector<Float, NX + 1, NGHOST> x{};
  Vector<Float, NX, NGHOST> xm{};
  Float dx{};

  Vector<Float, NY + 1, NGHOST> y{};
  Vector<Float, NY, NGHOST> ym{};
  Float dy{};

  Matrix<Float, NX, NY, NGHOST> visc{};

  Matrix<Float, NX, NY, NGHOST> p{};

  Float sigma{};                                      // Surface tension
  Matrix<Float, NX + 1, NY, NGHOST> p_jump_u_stag{};  // Pressure jump from surface tension
  Matrix<Float, NX, NY + 1, NGHOST> p_jump_v_stag{};  // Pressure jump from surface tension

  State<Float, NX, NY, NGHOST> old{};
  State<Float, NX, NY, NGHOST> curr{};
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void init_grid(Float x_min,
                         Float x_max,
                         Index nx,
                         Float y_min,
                         Float y_max,
                         Index ny,
                         FS<Float, NX, NY, NGHOST>& fs) noexcept {
  fs.dx = (x_max - x_min) / static_cast<Float>(nx);
  fs.dy = (y_max - y_min) / static_cast<Float>(ny);

  for_each_a<Exec::Parallel>(fs.x,
                             [&](Index i) { fs.x[i] = x_min + static_cast<Float>(i) * fs.dx; });
  for_each_a<Exec::Parallel>(fs.y,
                             [&](Index j) { fs.y[j] = y_min + static_cast<Float>(j) * fs.dy; });

  for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { fs.xm[i] = (fs.x[i] + fs.x[i + 1]) / 2; });
  for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { fs.ym[j] = (fs.y[j] + fs.y[j + 1]) / 2; });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void save_old_velocity(const State<Float, NX, NY, NGHOST>& curr,
                                 State<Float, NX, NY, NGHOST>& old) noexcept {
  std::copy_n(curr.U.get_data(), curr.U.size(), old.U.get_data());
  std::copy_n(curr.V.get_data(), curr.V.size(), old.V.get_data());
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void save_old_density(const State<Float, NX, NY, NGHOST>& curr,
                                State<Float, NX, NY, NGHOST>& old) noexcept {
  std::copy_n(curr.rho_u_stag.get_data(), curr.rho_u_stag.size(), old.rho_u_stag.get_data());
  std::copy_n(curr.rho_v_stag.get_data(), curr.rho_v_stag.size(), old.rho_v_stag.get_data());
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void save_old_state(const State<Float, NX, NY, NGHOST>& curr,
                              State<Float, NX, NY, NGHOST>& old) noexcept {
  save_old_density(curr, old);
  save_old_velocity(curr, old);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
auto adjust_dt(const FS<Float, NX, NY, NGHOST>& fs, Float cfl_max, Float dt_max) -> Float {
  Float CFLc_x = 0.0;
  Float CFLc_y = 0.0;
  Float CFLv_x = 0.0;
  Float CFLv_y = 0.0;

  for_each_i(fs.visc, [&](Index i, Index j) {
    CFLc_x         = std::max(CFLc_x, (fs.curr.U[i, j] + fs.curr.U[i + 1, j]) / 2 / fs.dx);
    CFLc_y         = std::max(CFLc_y, (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2 / fs.dy);

    const auto rho = (fs.curr.rho_u_stag[i, j] + fs.curr.rho_u_stag[i + 1, j] +
                      fs.curr.rho_v_stag[i, j] + fs.curr.rho_v_stag[i, j + 1]) /
                     4.0;
    CFLv_x = std::max(CFLv_x, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dx) * rho));
    CFLv_y = std::max(CFLv_y, 4.0 * fs.visc[i, j] / (Igor::sqr(fs.dy) * rho));
  });

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

template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto calc_rho_eps(const FS<Float, NX, NY, NGHOST>& fs) noexcept -> Float {
  return 1e-3 * std::min(fs.rho_gas, fs.rho_liquid);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_dmomdt(const FS<Float, NX, NY, NGHOST>& fs,
                 Matrix<Float, NX + 1, NY, NGHOST>& dmomUdt,
                 Matrix<Float, NX, NY + 1, NGHOST>& dmomVdt) {
  static Matrix<Float, NX, NY, 1> FXU{};
  static Matrix<Float, NX + 1, NY + 1, 0> FYU{};
  fill(FXU, std::numeric_limits<Float>::quiet_NaN());  // fill(FXU, 0.0);
  fill(FYU, std::numeric_limits<Float>::quiet_NaN());  // fill(FYU, 0.0);

  static Matrix<Float, NX + 1, NY + 1, 0> FXV{};
  static Matrix<Float, NX, NY, 1> FYV{};
  fill(FXV, std::numeric_limits<Float>::quiet_NaN());  // fill(FXV, 0.0);
  fill(FYV, std::numeric_limits<Float>::quiet_NaN());  // fill(FYV, 0.0);

  fill(dmomUdt, 0.0);
  fill(dmomVdt, 0.0);

  const auto rho_eps = calc_rho_eps(fs);

  // = Calculate dmomUdt ===========================================================================

  // = On center mesh ========================
  // FXU = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
  //     = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
  //     = -rho*U^2 + 2*mu*dUdx - p
  for_each<-1, NX + 1, 0, NY, Exec::Parallel>([&](Index i, Index j) {
    const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_u_stag[i, j],
                                                          fs.old.rho_u_stag[i + 1, j],
                                                          fs.curr.U[i, j],
                                                          fs.curr.U[i + 1, j],
                                                          fs.curr.U[i, j],
                                                          fs.curr.U[i + 1, j]);
    const auto U_i                        = ((fs.curr.U[i + 1, j] + fs.curr.U[i, j]) / 2);
    const auto dUdx                       = (fs.curr.U[i + 1, j] - fs.curr.U[i, j]) / fs.dx;

    FXU[i, j] = -rho_i_hybrid * U_i_hybrid * U_i + 2.0 * fs.visc[i, j] * dUdx - fs.p[i, j];
  });

  // = On corner mesh ======================
  // FYU = -rho*U*V + mu*(dUdy + dVdx)
  for_each_i<Exec::Parallel>(FYU, [&](Index i, Index j) {
    const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_u_stag[i, j - 1],
                                                          fs.old.rho_u_stag[i, j],
                                                          fs.curr.U[i, j - 1],
                                                          fs.curr.U[i, j],
                                                          fs.curr.V[i - 1, j],
                                                          fs.curr.V[i, j]);
    const auto V_i                        = (fs.curr.V[i - 1, j] + fs.curr.V[i, j]) / 2;

    const auto visc_corner =
        (fs.visc[i, j] + fs.visc[i - 1, j] + fs.visc[i, j - 1] + fs.visc[i - 1, j - 1]) / 4.0;
    const auto dUdy = (fs.curr.U[i, j] - fs.curr.U[i, j - 1]) / fs.dy;
    const auto dVdx = (fs.curr.V[i, j] - fs.curr.V[i - 1, j]) / fs.dx;

    FYU[i, j]       = -rho_i_hybrid * U_i_hybrid * V_i + visc_corner * (dUdy + dVdx);
  });

  for_each_i<Exec::Parallel>(dmomUdt, [&](Index i, Index j) {
    dmomUdt[i, j] = (FXU[i, j] - FXU[i - 1, j]) / fs.dx +  //
                    (FYU[i, j + 1] - FYU[i, j]) / fs.dy +  //
                    fs.p_jump_u_stag[i, j];
  });

  // = Calculate dmomVdt ===========================================================================

  // = On corner mesh ======================
  // FXV = -rho*U*V + mu*(dVdx + dUdy)
  for_each_i<Exec::Parallel>(FXV, [&](Index i, Index j) {
    const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_v_stag[i - 1, j],
                                                          fs.old.rho_v_stag[i, j],
                                                          fs.curr.V[i - 1, j],
                                                          fs.curr.V[i, j],
                                                          fs.curr.U[i, j - 1],
                                                          fs.curr.U[i, j]);
    const auto U_i                        = (fs.curr.U[i, j] + fs.curr.U[i, j - 1]) / 2;

    const auto visc_corner =
        (fs.visc[i, j] + fs.visc[i - 1, j] + fs.visc[i, j - 1] + fs.visc[i - 1, j - 1]) / 4.0;
    const auto dUdy = (fs.curr.U[i, j] - fs.curr.U[i, j - 1]) / fs.dy;
    const auto dVdx = (fs.curr.V[i, j] - fs.curr.V[i - 1, j]) / fs.dx;

    FXV[i, j]       = -rho_i_hybrid * U_i * V_i_hybrid + visc_corner * (dUdy + dVdx);
  });

  // = On center mesh ========================
  // FYV = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
  //     = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
  //     = -rho*V^2 + 2*mu*dVdy - p
  for_each<0, NX, -1, NY + 1, Exec::Parallel>([&](Index i, Index j) {
    const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_v_stag[i, j],
                                                          fs.old.rho_v_stag[i, j + 1],
                                                          fs.curr.V[i, j],
                                                          fs.curr.V[i, j + 1],
                                                          fs.curr.V[i, j],
                                                          fs.curr.V[i, j + 1]);
    const auto V_i                        = (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2;

    const auto dVdy                       = (fs.curr.V[i, j + 1] - fs.curr.V[i, j]) / fs.dy;

    FYV[i, j] = -rho_i_hybrid * V_i_hybrid * V_i + 2.0 * fs.visc[i, j] * dVdy - fs.p[i, j];
  });

  for_each_i<Exec::Parallel>(dmomVdt, [&](Index i, Index j) {
    dmomVdt[i, j] = (FXV[i + 1, j] - FXV[i, j]) / fs.dx +  //
                    (FYV[i, j] - FYV[i, j - 1]) / fs.dy +  //
                    fs.p_jump_v_stag[i, j];
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_drhodt(const FS<Float, NX, NY, NGHOST>& fs,
                 Matrix<Float, NX + 1, NY, NGHOST>& drho_u_stagdt,
                 Matrix<Float, NX, NY + 1, NGHOST>& drho_v_stagdt) {
  static Matrix<Float, NX, NY, 1> FXU{};
  static Matrix<Float, NX + 1, NY + 1, 0> FYU{};
  fill(FXU, std::numeric_limits<Float>::quiet_NaN());  // fill(FXU, 0.0);
  fill(FYU, std::numeric_limits<Float>::quiet_NaN());  // fill(FYU, 0.0);

  static Matrix<Float, NX + 1, NY + 1, 0> FXV{};
  static Matrix<Float, NX, NY, 1> FYV{};
  fill(FXV, std::numeric_limits<Float>::quiet_NaN());  // fill(FXV, 0.0);
  fill(FYV, std::numeric_limits<Float>::quiet_NaN());  // fill(FYV, 0.0);

  fill(drho_u_stagdt, 0.0);
  fill(drho_v_stagdt, 0.0);

  const auto rho_eps = calc_rho_eps(fs);

  // = Calculate drhodt for U-staggered density ====================================================

  // = On center mesh ========================
  // FXU = -rho * U
  for_each<-1, NX + 1, 0, NY, Exec::Parallel>([&](Index i, Index j) {
    const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                 fs.old.rho_u_stag[i, j],
                                                 fs.old.rho_u_stag[i + 1, j],
                                                 0.0,
                                                 0.0,
                                                 fs.curr.U[i, j],
                                                 fs.curr.U[i + 1, j]);
    const auto U_i               = (fs.curr.U[i, j] + fs.curr.U[i + 1, j]) / 2.0;
    FXU[i, j]                    = -rho_i_hybrid * U_i;
  });

  // = On corner mesh ======================
  // FYU = -rho * V
  for_each_i<Exec::Parallel>(FYU, [&](Index i, Index j) {
    const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                 fs.old.rho_u_stag[i, j - 1],
                                                 fs.old.rho_u_stag[i, j],
                                                 0.0,
                                                 0.0,
                                                 fs.curr.V[i - 1, j],
                                                 fs.curr.V[i, j]);
    const auto V_i               = (fs.curr.V[i - 1, j] + fs.curr.V[i, j]) / 2;
    FYU[i, j]                    = -rho_i_hybrid * V_i;
  });

  for_each_i<Exec::Parallel>(drho_u_stagdt, [&](Index i, Index j) {
    drho_u_stagdt[i, j] = (FXU[i, j] - FXU[i - 1, j]) / fs.dx +  //
                          (FYU[i, j + 1] - FYU[i, j]) / fs.dy;
  });

  // = Calculate drhodt for V-staggered density ====================================================

  // = On corner mesh ======================
  // FXV = -rho*U
  for_each_i<Exec::Parallel>(FXV, [&](Index i, Index j) {
    const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                 fs.old.rho_v_stag[i - 1, j],
                                                 fs.old.rho_v_stag[i, j],
                                                 0.0,
                                                 0.0,
                                                 fs.curr.U[i, j - 1],
                                                 fs.curr.U[i, j]);
    const auto U_i               = (fs.curr.U[i, j - 1] + fs.curr.U[i, j]) / 2.0;
    FXV[i, j]                    = -rho_i_hybrid * U_i;
  });

  // = On center mesh ========================
  // FYV = -rho*V
  for_each<0, NX, -1, NY + 1, Exec::Parallel>([&](Index i, Index j) {
    const auto [rho_i_hybrid, _] = hybrid_interp(rho_eps,
                                                 fs.old.rho_v_stag[i, j],
                                                 fs.old.rho_v_stag[i, j + 1],
                                                 0.0,
                                                 0.0,
                                                 fs.curr.V[i, j],
                                                 fs.curr.V[i, j + 1]);
    const auto V_i               = (fs.curr.V[i, j] + fs.curr.V[i, j + 1]) / 2.0;
    FYV[i, j]                    = -rho_i_hybrid * V_i;
  });

  for_each_i<Exec::Parallel>(drho_v_stagdt, [&](Index i, Index j) {
    drho_v_stagdt[i, j] = (FXV[i + 1, j] - FXV[i, j]) / fs.dx +  //
                          (FYV[i, j] - FYV[i, j - 1]) / fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_pressure_jump(const Matrix<Float, NX, NY, NGHOST>& vf,
                        const Matrix<Float, NX, NY, NGHOST>& curv,
                        FS<Float, NX, NY, NGHOST>& fs) noexcept {
  fill(fs.p_jump_u_stag, 0.0);
  fill(fs.p_jump_v_stag, 0.0);

  for_each_i<Exec::Parallel>(fs.p_jump_u_stag, [&](Index i, Index j) {
    const auto minus_has_interface = static_cast<Float>(has_interface(vf, i - 1, j));
    const auto plus_has_interface  = static_cast<Float>(has_interface(vf, i, j));
    const auto curv_i =
        (plus_has_interface + minus_has_interface) > 0.0
            ? (curv[i, j] * plus_has_interface + curv[i - 1, j] * minus_has_interface) /
                  (plus_has_interface + minus_has_interface)
            : 0.0;
    fs.p_jump_u_stag[i, j] = fs.sigma * curv_i * (vf[i, j] - vf[i - 1, j]) / fs.dx;
  });

  for_each_i<Exec::Parallel>(fs.p_jump_v_stag, [&](Index i, Index j) {
    const auto minus_has_interface = static_cast<Float>(has_interface(vf, i, j - 1));
    const auto plus_has_interface  = static_cast<Float>(has_interface(vf, i, j));
    const auto curv_i =
        (plus_has_interface + minus_has_interface) > 0.0
            ? (curv[i, j] * plus_has_interface + curv[i, j - 1] * minus_has_interface) /
                  (plus_has_interface + minus_has_interface)
            : 0.0;
    fs.p_jump_v_stag[i, j] = fs.sigma * curv_i * (vf[i, j] - vf[i, j - 1]) / fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void calc_rho_and_visc(FS<Float, NX, NY, NGHOST>& fs) noexcept {
  IGOR_ASSERT(std::abs(fs.rho_gas - fs.rho_liquid) < 1e-12,
              "Expected constant density but rho_gas = {:.6e} and rho_liquid = {:.6e}",
              fs.rho_gas,
              fs.rho_liquid);
  IGOR_ASSERT(std::abs(fs.visc_gas - fs.visc_liquid) < 1e-12,
              "Expected constant viscosity but visc_gas = {:.6e} and visc_liquid = {:.6e}",
              fs.visc_gas,
              fs.visc_liquid);

  fill(fs.old.rho_u_stag, fs.rho_gas);
  fill(fs.old.rho_v_stag, fs.rho_gas);
  fill(fs.curr.rho_u_stag, fs.rho_gas);
  fill(fs.curr.rho_v_stag, fs.rho_gas);
  fill(fs.visc, fs.visc_gas);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void calc_rho_and_visc(const Matrix<Float, NX, NY, NGHOST>& vf,
                                 FS<Float, NX, NY, NGHOST>& fs) noexcept {
  // = Density on U-staggered mesh =================================================================
  for_each_i<Exec::Parallel>(fs.curr.rho_u_stag, [&](Index i, Index j) {
    const auto rho_minus     = vf[i - 1, j] * fs.rho_liquid + (1.0 - vf[i - 1, j]) * fs.rho_gas;
    const auto rho_plus      = vf[i, j] * fs.rho_liquid + (1.0 - vf[i, j]) * fs.rho_gas;
    fs.curr.rho_u_stag[i, j] = (rho_minus + rho_plus) / 2.0;
  });
  apply_neumann_bconds(fs.curr.rho_u_stag);

  // = Density on V-staggered mesh =================================================================
  for_each_i<Exec::Parallel>(fs.curr.rho_v_stag, [&](Index i, Index j) {
    const auto rho_minus     = vf[i, j - 1] * fs.rho_liquid + (1.0 - vf[i, j - 1]) * fs.rho_gas;
    const auto rho_plus      = vf[i, j] * fs.rho_liquid + (1.0 - vf[i, j]) * fs.rho_gas;
    fs.curr.rho_v_stag[i, j] = (rho_minus + rho_plus) / 2.0;
  });
  apply_neumann_bconds(fs.curr.rho_v_stag);

  // = Viscosity on centered mesh ==================================================================
  for_each_i<Exec::Parallel>(fs.visc, [&](Index i, Index j) {
    fs.visc[i, j] = vf[i, j] * fs.visc_liquid + (1.0 - vf[i, j]) * fs.visc_gas;
  });
  apply_neumann_bconds(fs.visc);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_conserved_quantities(const FS<Float, NX, NY, NGHOST>& fs,
                               Float& mass,
                               Float& momentum_x,
                               Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      mass += (fs.curr.rho_u_stag[i, j] + fs.curr.rho_u_stag[i + 1, j] + fs.curr.rho_v_stag[i, j] +
               fs.curr.rho_v_stag[i, j + 1]) /
              4.0 * fs.dx * fs.dy;

      momentum_x += (fs.curr.rho_u_stag[i, j] * fs.curr.U[i, j] +
                     fs.curr.rho_u_stag[i + 1, j] * fs.curr.U[i + 1, j]) /
                    2.0 * fs.dx * fs.dy;
      momentum_y += (fs.curr.rho_v_stag[i, j] * fs.curr.V[i, j] +
                     fs.curr.rho_v_stag[i, j + 1] * fs.curr.V[i, j + 1]) /
                    2.0 * fs.dx * fs.dy;
    }
  }
}

#endif  // FLUID_SOLVER_FS_HPP_
