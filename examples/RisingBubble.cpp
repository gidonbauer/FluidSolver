#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
// #define FS_SILENCE_CONV_WARN
// #define FS_VOF_ADVECT_WITH_STAGGERED_VELOCITY

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
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = VTKWriter<Float, NX, NY, NGHOST>;
#else
#include "XDMFWriter.hpp"
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = XDMFWriter<Float, NX, NY, NGHOST>;
#endif  // USE_VTK

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 256;
constexpr Index NY              = 256;
constexpr Index NGHOST          = 1;

constexpr Float SCALE           = 0.25;
constexpr Float X_MIN           = -1.0 * SCALE;
constexpr Float X_MAX           = 1.0 * SCALE;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 2.0 * SCALE;

constexpr Float T_END           = 5.0;
constexpr Float DT_MAX          = 1e-3;
constexpr Float CFL_MAX         = 0.25;
constexpr Float DT_WRITE        = 1e-2;

constexpr Float V_IN            = 0.0;
constexpr Float GRAVITY         = -1e-1;

constexpr Float VISC_G          = 1e-6;  // 1e-4;
constexpr Float RHO_G           = 1e0;   // 8e-2;
constexpr Float VISC_L          = 1e-2;  // 1e-7;
constexpr Float RHO_L           = 1e3;   // 1e3;

constexpr Float SURFACE_TENSION = 1.0 / 20.0;
constexpr Float CX              = 0.0;
constexpr Float CY              = 0.25 * SCALE;
constexpr Float R0              = 0.125 * SCALE;

constexpr int PRESSURE_MAX_ITER = 100;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Eötvös number
constexpr Float Eo = RHO_L * Igor::abs(GRAVITY) * Igor::sqr(R0) / SURFACE_TENSION;
// constexpr Float Eo = (RHO_L - RHO_G) * Igor::abs(GRAVITY) * Igor::sqr(2.0 * R0) /
// SURFACE_TENSION;
// Galilei number
constexpr Float Ga = Igor::sqrt((RHO_L * Igor::abs(GRAVITY) * R0 * R0 * R0) / Igor::sqr(VISC_L));
// Density ratio
constexpr Float Rho_r = RHO_L / RHO_G;
// Viscosity ratio
constexpr Float Visc_r = VISC_L / VISC_G;

// Weber number
constexpr Float We = RHO_G * Igor::sqr(V_IN) * 2.0 * R0 / SURFACE_TENSION;
// Morton number
constexpr Float Mo = Igor::abs(GRAVITY) * Igor::sqr(Igor::sqr(VISC_G)) * (RHO_L - RHO_G) /
                     (Igor::sqr(RHO_G) * Igor::sqr(SURFACE_TENSION) * SURFACE_TENSION);
// See: Mechanism study of bubble dynamics under the buoyancy effects, Huang
constexpr Float Bu = (RHO_L - RHO_G) / RHO_G;

constexpr FlowBConds<Float> bconds{
    .left   = Neumann{},
    .right  = Neumann{},
    .bottom = Dirichlet{.U = 0.0, .V = V_IN},
    .top    = Neumann{},
};

#ifndef FS_BASE_DIR
#define FS_BASE_DIR "."
#endif  // FS_BASE_DIR
constexpr auto OUTPUT_DIR = FS_BASE_DIR "/output/RisingBubble/";
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
  for_each_a(fs.xm, [&](Index i) {
    inflow  += fs.curr.rho_v_stag(i, -NGHOST) * fs.curr.V(i, -NGHOST);
    outflow += fs.curr.rho_v_stag(i, NY + NGHOST) * fs.curr.V(i, NY + NGHOST);
  });
  mass_error = outflow - inflow;
}

// -------------------------------------------------------------------------------------------------
auto calc_center_of_mass(Float dx,
                         Float dy,
                         const Vector<Float, NX, NGHOST>& xm,
                         const Vector<Float, NY, NGHOST>& ym,
                         const Matrix<Float, NX, NY, NGHOST>& vf) -> std::array<Float, 2> {
  const auto vol   = integrate(dx, dy, vf);

  Float weighted_x = 0.0;
  Float weighted_y = 0.0;
  for_each_i(vf, [&](Index i, Index j) {
    weighted_x += xm(i) * vf(i, j);
    weighted_y += ym(j) * vf(i, j);
  });
  weighted_x *= dx * dy;
  weighted_y *= dx * dy;

  return {weighted_x / vol, weighted_y / vol};
}
// -------------------------------------------------------------------------------------------------
auto main(int argc, char** argv) -> int {
  Igor::Info("Eo = {:.6e}", Eo);
  Igor::Info("Ga = {:.6e}", Ga);
  Igor::Info("rho ratio  = {:.6e}", Rho_r);
  Igor::Info("visc ratio = {:.6e}", Visc_r);
  std::cout << '\n';
  Igor::Info("Mo = {:.6e}", Mo);
  Igor::Info("Bu = {:.6e}", Bu);
  Igor::Info("We = {:.6e}", We);

  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{.visc_gas    = VISC_L,
                               .visc_liquid = VISC_G,
                               .rho_gas     = RHO_L,
                               .rho_liquid  = RHO_G,
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
  Float p_res                  = 0.0;
  Index p_iter                 = 0;

  [[maybe_unused]] Float com_x = 0.0;
  Float com_y                  = 0.0;
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

  // monitor.add_variable(&We, "We");
  // monitor.add_variable(&Eo, "Eo");
  // monitor.add_variable(&Mo, "Mo");
  // monitor.add_variable(&Bu, "Bu");

  // monitor.add_variable(&com_x, "com_x");
  monitor.add_variable(&com_y, "com_y");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  const int vof0_config = [argc, argv]() {
    auto usage = [prog = argv[0]]() {
      std::cerr << "Usage: " << prog << " [bubble config]\n";
      std::cerr << "       bubble config:  0 - Single bubble (default)\n";
      std::cerr << "                       1 - Two bubbles side by side\n";
      std::cerr << "                       2 - Two bubbles above each other\n";
    };

    if (argc < 2) {
      usage();
      return 0;
    }
    switch (argv[1][0]) {
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      default:  usage(); return 0;
    }
  }();

  auto vof0 = [vof0_config](Float x, Float y) -> Float {
    switch (vof0_config) {
      // Single bubble
      case 0: return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));

      // Two bubbles side by side
      case 1:
        return static_cast<Float>(
            Igor::sqr(x - (CX - 2.0 * R0)) + Igor::sqr(y - CY) <= Igor::sqr(R0) ||
            Igor::sqr(x - (CX + 2.0 * R0)) + Igor::sqr(y - CY) <= Igor::sqr(R0));

      // Two bubbles above each other
      case 2:
        return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0) ||
                                  Igor::sqr(x - CX) + Igor::sqr(y - (CY + 3.0 * R0)) <=
                                      Igor::sqr(R0));

      default: Igor::Panic("Unreachable: Invalid vof0_config = {}", vof0_config);
    }
    std::unreachable();
  };

  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf(i, j) = quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  const Float init_vf_integral = integrate<true>(fs.dx, fs.dy, vof.vf);
  localize_cells(fs.x, fs.y, vof.ir);
  reconstruct_interface(fs, vof.vf, vof.ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  calc_rho(vof.vf, fs);
  calc_visc(vof.vf, fs);
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::NONE);

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
  auto com = calc_center_of_mass(fs.dx, fs.dy, fs.xm, fs.ym, vof.vf);
  com_x    = com[0];
  com_y    = com[1];
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);
    std::copy_n(vof.vf.get_data(), vof.vf.size(), vof.vf_old.get_data());

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs, vof.vf_old, vof.ir);
    // calc_surface_length(fs, ir, interface_length);
    // TODO: Calculate viscosity from new VOF field
    calc_rho(vof.vf_old, fs);
    save_old_density(fs.curr, fs.old);

    advect_cells(fs, Ui, Vi, dt, vof, &vof_vol_error);
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

      // Correct the outflow
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
        fs.curr.V(i, NY + NGHOST) -=
            mass_error / (fs.curr.rho_v_stag(i, NY + NGHOST) * static_cast<Float>(NX + 2 * NGHOST));
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
    U_max    = abs_max(fs.curr.U);
    V_max    = abs_max(fs.curr.V);
    div_max  = abs_max(div);
    curv_min = min(vof.curv);
    curv_max = max(vof.curv);
    // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
    // p_max = max(fs.p);
    calc_vof_stats(fs, vof.vf, init_vf_integral, vof_min, vof_max, vof_integral, vof_loss);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    com   = calc_center_of_mass(fs.dx, fs.dy, fs.xm, fs.ym, vof.vf);
    com_x = com[0];
    com_y = com[1];
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
