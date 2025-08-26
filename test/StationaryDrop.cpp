#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
// #define FS_VOF_ADVECT_WITH_STAGGERED_VELOCITY

#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "XDMFWriter.hpp"

// TODO: Test case for capillary forces: Stationary drop, no flow -> capillary forces should not
//       induce a current. Important: only calculate a quarter of the drop and use Neumann boundary
//       conditions.

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 64;
constexpr Index NY              = 64;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 1.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 60.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-1;

constexpr Float VISC_G          = 1e-3;  // 1e-0;
constexpr Float RHO_G           = 1.0;
constexpr Float VISC_L          = 1e-3;
constexpr Float RHO_L           = 1e3;

constexpr Float SURFACE_TENSION = 1.0 / 20.0;
constexpr Float CX              = 0.5;
constexpr Float CY              = 0.5;
constexpr Float R0              = 0.25;
constexpr auto vof0             = [](Float x, Float y) {
  return static_cast<Float>(Igor::sqr(2.0 * (x - CX)) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 3;

#if 0
constexpr FlowBConds<Float> bconds{
    //        LEFT             RIGHT            BOTTOM           TOP
    .types = {BCond::SYMMETRY, BCond::SYMMETRY, BCond::SYMMETRY, BCond::SYMMETRY},
    .U     = {0.0, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};
#else
constexpr FlowBConds<Float> bconds{
    //        LEFT            RIGHT           BOTTOM          TOP
    .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::NEUMANN, BCond::NEUMANN},
    .U     = {0.0, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};
#endif

constexpr auto OUTPUT_DIR = "test/output/StationaryDrop/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_vof_stats(const FS<Float, NX, NY, NGHOST>& fs,
                    const Matrix<Float, NX, NY, NGHOST>& vf,
                    const Float init_vf_integral,
                    Float& min,
                    Float& max,
                    Float& integral,
                    Float& loss) noexcept {
  const auto [min_it, max_it] = std::minmax_element(vf.get_data(), vf.get_data() + vf.size());

  min                         = *min_it;
  max                         = *max_it;
  integral                    = integrate<true>(fs.dx, fs.dy, vf);
  loss                        = init_vf_integral - integral;
}

// -------------------------------------------------------------------------------------------------
#if 0
auto check_vf_ghost(const FS<Float, NX, NY, NGHOST>& fs, const Matrix<Float, NX, NY, NGHOST>& vf) {
  bool left_good = true;
  for_each_a(fs.ym, [&](Index j) {
    if (fs.y(j + 1) < R0) {
      left_good = left_good && std::abs(vf(-NGHOST, j) - 1.0) < 1e-8;
      left_good = left_good && std::abs(vf(0, j) - 1.0) < 1e-8;
    }
  });
  if (!left_good) { Igor::Warn("Error in vf on left side"); }

  bool bottom_good = true;
  for_each_a(fs.xm, [&](Index i) {
    if (fs.x(i + 1) < R0) {
      bottom_good = bottom_good && std::abs(vf(i, -NGHOST) - 1.0) < 1e-8;
      bottom_good = bottom_good && std::abs(vf(i, 0) - 1.0) < 1e-8;
    }
  });
  if (!bottom_good) { Igor::Warn("Error in vf on bottom side"); }

  return left_good && bottom_good;
}
#endif

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{.visc_gas    = VISC_G,
                               .visc_liquid = VISC_L,
                               .rho_gas     = RHO_G,
                               .rho_liquid  = RHO_L,
                               .sigma       = SURFACE_TENSION};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  VOF<Float, NX, NY, NGHOST> vof{};

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};
  Matrix<Float, NX, NY, NGHOST> rhoi{};

  Matrix<Float, NX + 1, NY, NGHOST> drho_u_stagdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drho_v_stagdt{};
  Matrix<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Matrix<Float, NX, NY, NGHOST> delta_p{};
  Matrix<Float, NX + 1, NY, NGHOST> delta_pj_u_stag{};
  Matrix<Float, NX, NY + 1, NGHOST> delta_pj_v_stag{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float mass    = 0.0;
  Float mom_x   = 0.0;
  Float mom_y   = 0.0;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;
  // Float div_L1        = 0.0;

  Float vof_min       = 0.0;
  Float vof_max       = 0.0;
  Float vof_integral  = 0.0;
  Float vof_loss      = 0.0;
  Float vof_vol_error = 0.0;

  // Float p_max         = 0.0;
  Float p_res  = 0.0;
  Index p_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  XDMFWriter<Float, NX, NY, NGHOST> data_writer(
      Igor::detail::format("{}/solution.xdmf2", OUTPUT_DIR),
      Igor::detail::format("{}/solution.h5", OUTPUT_DIR),
      &fs.x,
      &fs.y);
  data_writer.add_scalar("density", &rhoi);
  data_writer.add_scalar("viscosity", &fs.visc);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_scalar("VOF", &vof.vf);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("curvature", &vof.curv);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");
  // monitor.add_variable(&div_L1, "L1(div)");

  // monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  // monitor.add_variable(&vof_integral, "int(vof)");
  monitor.add_variable(&vof_loss, "loss(vof)");
  // monitor.add_variable(&vof_vol_error, "max(vol. error)");

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf(i, j) = quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  apply_neumann_bconds(vof.vf);
#if 0 
  IGOR_ASSERT(check_vf_ghost(fs, vof.vf), "Error in vf.");
#endif
  const Float init_vf_integral = integrate<true>(fs.dx, fs.dy, vof.vf);
  localize_cells(fs.x, fs.y, vof.ir);
  reconstruct_interface(fs, vof.vf, vof.ir);
  // = Initialize VOF field ========================================================================

  calc_rho_and_visc(vof.vf, fs);
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::RIGHT);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
  // p_max = abs_max(fs.p);
  calc_vof_stats(fs, vof.vf, init_vf_integral, vof_min, vof_max, vof_integral, vof_loss);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("StationaryDrop");
  bool any_test_failed = false;
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);
    copy(vof.vf, vof.vf_old);
#if 0
    IGOR_ASSERT(check_vf_ghost(fs, vof.vf_old), "Error in vf_old at time t={:.6e}", t);
#endif

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs, vof.vf_old, vof.ir);
    // TODO: Calculate viscosity from new VOF field
    calc_rho_and_visc(vof.vf_old, fs);
    save_old_density(fs.curr, fs.old);

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    advect_cells(fs, Ui, Vi, dt, vof, &vof_vol_error);
    apply_neumann_bconds(vof.vf);
#if 0
    IGOR_ASSERT(check_vf_ghost(fs, vof.vf), "Error in vf at time t={:.6e}", t);
#endif

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update the density field to make the update consistent ==================================
      calc_drhodt(fs, drho_u_stagdt, drho_v_stagdt);
      IGOR_ASSERT(std::none_of(drho_u_stagdt.get_data(),
                               drho_u_stagdt.get_data() + drho_u_stagdt.size(),
                               [](Float x) { return std::isnan(x); }),
                  "NaN value in drho_u_stagdt.");
      IGOR_ASSERT(std::none_of(drho_v_stagdt.get_data(),
                               drho_v_stagdt.get_data() + drho_v_stagdt.size(),
                               [](Float x) { return std::isnan(x); }),
                  "NaN value in drho_v_stagdt.");

      for_each_i<Exec::Parallel>(fs.curr.rho_u_stag, [&](Index i, Index j) {
        fs.curr.rho_u_stag(i, j) = fs.old.rho_u_stag(i, j) + dt * drho_u_stagdt(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.rho_v_stag, [&](Index i, Index j) {
        fs.curr.rho_v_stag(i, j) = fs.old.rho_v_stag(i, j) + dt * drho_v_stagdt(i, j);
      });
      apply_neumann_bconds(fs.curr.rho_u_stag);
      apply_neumann_bconds(fs.curr.rho_v_stag);
      if (std::any_of(fs.curr.rho_u_stag.get_data(),
                      fs.curr.rho_u_stag.get_data() + fs.curr.rho_u_stag.size(),
                      [](Float x) { return std::abs(x) < 1e-8; })) {
        Igor::Warn("t={}, subiter={}: Zero in rho_u_stag", t, sub_iter);
        return 1;
      }
      if (std::any_of(fs.curr.rho_v_stag.get_data(),
                      fs.curr.rho_v_stag.get_data() + fs.curr.rho_v_stag.size(),
                      [](Float x) { return std::abs(x) < 1e-8; })) {
        Igor::Warn("t={}, subiter={}: Zero in rho_v_stag", t, sub_iter);
        return 1;
      }
      ps.setup(fs);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });
      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      // ===== Add capillary forces ================================================================
      // calc_curvature_convolved_vf(fs, vof);
      // calc_curvature_quad_regression(fs, vof);
      calc_curvature_quad_volume_matching(fs, vof);
      apply_neumann_bconds(vof.curv);
      if (std::any_of(vof.curv.get_data(), vof.curv.get_data() + vof.curv.size(), [](Float x) {
            return std::isnan(x);
          })) {
        Igor::Warn("t={}, subiter={}: NaN value in curvature.", t, sub_iter);
        return 1;
      }

      // NOTE: Save old pressure jump in delta_pj_[uv]_stag
      copy(fs.p_jump_u_stag, delta_pj_u_stag);
      copy(fs.p_jump_v_stag, delta_pj_v_stag);
      calc_interface_length(fs, vof);
      calc_pressure_jump(vof.vf_old, vof.curv, vof.interface_length, fs);
      apply_neumann_bconds(fs.p_jump_u_stag);
      apply_neumann_bconds(fs.p_jump_v_stag);
      for_each_a<Exec::Parallel>(delta_pj_u_stag, [&](Index i, Index j) {
        delta_pj_u_stag(i, j) = fs.p_jump_u_stag(i, j) - delta_pj_u_stag(i, j);
      });
      for_each_a<Exec::Parallel>(delta_pj_v_stag, [&](Index i, Index j) {
        delta_pj_v_stag(i, j) = fs.p_jump_v_stag(i, j) - delta_pj_v_stag(i, j);
      });

      for_each_a<Exec::Parallel>(div, [&](Index i, Index j) {
        div(i, j) += dt * ((delta_pj_u_stag(i + 1, j) / fs.curr.rho_u_stag(i + 1, j) -
                            delta_pj_u_stag(i, j) / fs.curr.rho_u_stag(i, j)) /
                               fs.dx +
                           (delta_pj_v_stag(i, j + 1) / fs.curr.rho_v_stag(i, j + 1) -
                            delta_pj_v_stag(i, j) / fs.curr.rho_v_stag(i, j)) /
                               fs.dy);
      });
      // ===== Add capillary forces ================================================================

      Index local_p_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
      }
      p_iter += local_p_iter;
      if (std::isnan(p_res) || std::any_of(delta_p.get_data(),
                                           delta_p.get_data() + delta_p.size(),
                                           [](Float x) { return std::isnan(x); })) {
        Igor::Warn("t={}, subiter={}: NaN value in pressure correction.", t, sub_iter);
        return 1;
      }

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      // Correct pressure
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      // Correct velocity
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        const auto dpdx  = (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx;
        const auto rho   = fs.curr.rho_u_stag(i, j);
        fs.curr.U(i, j) -= dpdx * dt / rho;
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        const auto dpdy  = (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy;
        const auto rho   = fs.curr.rho_v_stag(i, j);
        fs.curr.V(i, j) -= dpdy * dt / rho;
      });
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
    // p_max = abs_max(fs.p);
    calc_vof_stats(fs, vof.vf, init_vf_integral, vof_min, vof_max, vof_integral, vof_loss);

    {
      if (vof_min < -1e-8) {
        Igor::Warn(
            "Incorrect vf_min at time {:.6e}, expected {:.6e} but got {:.6e}", t, 0.0, vof_min);
        any_test_failed = true;
      }
      if (vof_max > 1.0 + 1e-8) {
        Igor::Warn(
            "Incorrect vf_max at time {:.6e}, expected {:.6e} but got {:.6e}", t, 1.0, vof_max);
        any_test_failed = true;
      }
      if (std::abs(vof_loss) > 1e-8) {
        Igor::Warn("High volume loss at time {:.6e}: {:.6e}", t, vof_loss);
        any_test_failed = true;
      }
    }

    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

#if 0
  // L1-error of last solution to initial condition
  std::atomic<Float> L1_error = 0.0;
  for_each_i<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    const auto expected =
        quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
    const auto actual  = vof.vf(i, j);
    L1_error          += std::abs(expected - actual);
  });
  L1_error = L1_error * fs.dx * fs.dy;
  Igor::Debug("L1 error final solution = {:.6e}", static_cast<Float>(L1_error));
#endif

  return any_test_failed ? 1 : 0;
}
