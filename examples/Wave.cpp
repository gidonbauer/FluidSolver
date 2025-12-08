#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN
// #define FS_ARITHMETIC_VISC

#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 5LL * 128;
constexpr Index NY              = 128;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 5.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 30.0;
constexpr Float DT_MAX          = 1e-1;  // 5e-4;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 5e-2;

constexpr Float RHO_G           = 1.0;
constexpr Float VISC_G          = 1e-6;
constexpr Float RHO_L           = 1e3;
constexpr Float VISC_L          = 1e-3;
constexpr Float GRAVITY         = -1e-0;

constexpr Float SURFACE_TENSION = 1.0 / 20.0;
constexpr auto vof0             = [](Float x, [[maybe_unused]] Float y) {
  return static_cast<Float>(y < 0.9 * std::exp(-Igor::sqr((x - 2.5) / 0.5)));
  // return static_cast<Float>(2.0 <= x && x <= 3.0);
  // return static_cast<Float>(2.0 <= x && x <= 3.0 && 0.1 <= y && y <= 0.9);
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .right  = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
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
  integral                    = integrate(fs.dx, fs.dy, vf);
  loss                        = init_vf_integral - integral;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
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
  Matrix<Float, NX + 1, NY, NGHOST> delta_p_jump_u_stag{};
  Matrix<Float, NX, NY + 1, NGHOST> delta_p_jump_v_stag{};

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

  Float curv_min      = 0.0;
  Float curv_max      = 0.0;

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
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
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

  monitor.add_variable(&curv_min, "min(curv)");
  monitor.add_variable(&curv_max, "max(curv)");

  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  // monitor.add_variable(&vof_integral, "int(vof)");
  monitor.add_variable(&vof_loss, "loss(vof)");
  // monitor.add_variable(&vof_vol_error, "max(vol. error)");

  // monitor.add_variable(&mass, "mass");
  // monitor.add_variable(&mom_x, "momentum (x)");
  // monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf(i, j) = quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  apply_neumann_bconds(vof.vf);
  const Float init_vf_integral = integrate(fs.dx, fs.dy, vof.vf);
  localize_cells(fs.x, fs.y, vof.ir);
  reconstruct_interface(fs, vof.vf, vof.ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  calc_rho(vof.vf, fs);
  calc_visc(vof.vf, fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max    = abs_max(fs.curr.U);
  V_max    = abs_max(fs.curr.V);
  div_max  = abs_max(div);
  curv_min = min(vof.curv);
  curv_max = max(vof.curv);
  // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
  // p_max = max(fs.p);
  calc_vof_stats(fs, vof.vf, init_vf_integral, vof_min, vof_max, vof_integral, vof_loss);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt                      = adjust_dt(fs, CFL_MAX, DT_MAX);
    const auto cfl_gravitiy = 1.0 / std::sqrt(fs.dy / std::abs(GRAVITY));
    const auto dt_gravitiy  = CFL_MAX / cfl_gravitiy;
    dt                      = std::min(dt, dt_gravitiy);
    dt                      = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);
    copy(vof.vf, vof.vf_old);

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs, vof.vf_old, vof.ir);
    calc_rho(vof.vf_old, fs);
    save_old_density(fs.curr, fs.old);

    advect_cells(fs, Ui, Vi, dt, vof, &vof_vol_error);
    apply_neumann_bconds(vof.vf);
    calc_visc(vof.vf, fs);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update the density field to make the update consistent ==================================
      calc_drhodt(fs, drho_u_stagdt, drho_v_stagdt);
      for_each_i<Exec::Parallel>(fs.curr.rho_u_stag, [&](Index i, Index j) {
        fs.curr.rho_u_stag(i, j) = fs.old.rho_u_stag(i, j) + dt * drho_u_stagdt(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.rho_v_stag, [&](Index i, Index j) {
        fs.curr.rho_v_stag(i, j) = fs.old.rho_v_stag(i, j) + dt * drho_v_stagdt(i, j);
      });
      apply_neumann_bconds(fs.curr.rho_u_stag);
      apply_neumann_bconds(fs.curr.rho_v_stag);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      // = Add Gravity =======
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        drhoVdt(i, j) += fs.curr.rho_v_stag(i, j) * GRAVITY;
      });

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
      calc_curvature_quad_volume_matching(fs, vof);

      // NOTE: Save old pressure jump in delta_p_jump_[uv]_stag
      copy(fs.p_jump_u_stag, delta_p_jump_u_stag);
      copy(fs.p_jump_v_stag, delta_p_jump_v_stag);
      calc_interface_length(fs, vof);
      calc_pressure_jump(vof.vf_old, vof.curv, vof.interface_length, fs);
      for_each_a<Exec::Parallel>(delta_p_jump_u_stag, [&](Index i, Index j) {
        delta_p_jump_u_stag(i, j) = fs.p_jump_u_stag(i, j) - delta_p_jump_u_stag(i, j);
      });
      for_each_a<Exec::Parallel>(delta_p_jump_v_stag, [&](Index i, Index j) {
        delta_p_jump_v_stag(i, j) = fs.p_jump_v_stag(i, j) - delta_p_jump_v_stag(i, j);
      });

      for_each_i<Exec::Parallel>(div, [&](Index i, Index j) {
        div(i, j) += dt * ((delta_p_jump_u_stag(i + 1, j) / fs.curr.rho_u_stag(i + 1, j) -
                            delta_p_jump_u_stag(i, j) / fs.curr.rho_u_stag(i, j)) /
                               fs.dx +
                           (delta_p_jump_v_stag(i, j + 1) / fs.curr.rho_v_stag(i, j + 1) -
                            delta_p_jump_v_stag(i, j) / fs.curr.rho_v_stag(i, j)) /
                               fs.dy);
      });
      // ===== Add capillary forces ================================================================

      Index local_p_iter = 0;
      ps.set_pressure_operator(fs);
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;
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
    U_max    = max(fs.curr.U);
    V_max    = max(fs.curr.V);
    div_max  = max(div);
    curv_min = min(vof.curv);
    curv_max = max(vof.curv);
    // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
    // p_max = max(fs.p);
    calc_vof_stats(fs, vof.vf, init_vf_integral, vof_min, vof_max, vof_integral, vof_loss);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
