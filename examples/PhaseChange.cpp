#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN

#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#if defined(USE_VTK) || defined(FS_DISABLE_HDF)
#include "VTKWriter.hpp"
template <typename Float, int NX, int NY, int NGHOST>
using DataWriter = VTKWriter<Float, NX, NY, NGHOST>;
#else
#include "XDMFWriter.hpp"
template <typename Float, int NX, int NY, int NGHOST>
using DataWriter = XDMFWriter<Float, NX, NY, NGHOST>;
#endif  // USE_VTK

// TODO: Test case for capillary forces: Stationary drop, no flow -> capillary forces should not
//       induce a current. Important: only calculate a quarter of the drop and use Neumann boundary
//       conditions.

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 5 * 32;
constexpr Index NY              = 32;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 5.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 1.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-8;

constexpr Float U_0             = 0.0;
constexpr Float VISC_G          = 1e-3;
constexpr Float RHO_G           = 1.0;
constexpr Float VISC_L          = 1e-3;
constexpr Float RHO_L           = 1e3;

constexpr Float SURFACE_TENSION = 1.0 / 20.0;
constexpr auto vof0             = [](Float x, [[maybe_unused]] Float y) { return static_cast<Float>(x > 1.0); };

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM          TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::NEUMANN, BCond::NEUMANN},
    .U     = {0.0, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "output/PhaseChange/";
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
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0.0;
  outflow = 0.0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag(-NGHOST, j) * fs.curr.U(-NGHOST, j);
    outflow += fs.curr.rho_u_stag(NX + NGHOST, j) * fs.curr.U(NX + NGHOST, j);
  });
  mass_error = outflow - inflow;
}

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

  Float curv_min     = 0.0;
  Float curv_max     = 0.0;

  Float vf_min       = 0.0;
  Float vf_max       = 0.0;
  Float vf_integral  = 0.0;
  Float vf_loss      = 0.0;
  Float vf_vol_error = 0.0;

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

  monitor.add_variable(&vf_min, "min(vof)");
  monitor.add_variable(&vf_max, "max(vof)");
  // monitor.add_variable(&vf_integral, "int(vof)");
  monitor.add_variable(&vf_loss, "loss(vof)");
  // monitor.add_variable(&vf_vol_error, "max(vol. error)");

  // monitor.add_variable(&mass, "mass");
  // monitor.add_variable(&mom_x, "momentum (x)");
  // monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf(i, j) = quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  const Float init_vf_integral = integrate<true>(fs.dx, fs.dy, vof.vf);
  localize_cells(fs.x, fs.y, vof.ir);
  reconstruct_interface(fs, vof.vf, vof.ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = U_0; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  calc_rho(vof.vf, fs);
  calc_visc(vof.vf, fs);
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::RIGHT);

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
  calc_vof_stats(fs, vof.vf, init_vf_integral, vf_min, vf_max, vf_integral, vf_loss);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);
    copy(vof.vf, vof.vf_old);

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs, vof.vf_old, vof.ir);
    // calc_surface_length(fs, ir, interface_length);
    // TODO: Calculate viscosity from new VOF field
    calc_rho(vof.vf_old, fs);
    save_old_density(fs.curr, fs.old);

    advect_cells(fs, Ui, Vi, dt, vof, &vf_vol_error);
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
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });
      apply_velocity_bconds(fs, bconds);

      // Correct the outflow
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });

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
      ps.setup(fs);
      ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter);
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
    calc_vof_stats(fs, vof.vf, init_vf_integral, vf_min, vf_max, vf_integral, vf_loss);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
