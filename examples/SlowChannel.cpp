#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "Curvature.hpp"
#include "FS.hpp"
#include "Geometry.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"

// = Config ========================================================================================
using Float                = double;

constexpr Float Re         = 1e-3;
constexpr Float We         = 1e-3;
constexpr Float rhor       = 1000.0;
constexpr Float mur        = 1000.0;

constexpr Float L          = 2.0;
constexpr Float D          = 0.25 * L;
constexpr Float RHO_L      = 1.0;
constexpr Float MU_L       = 1e-3;

constexpr Float U_MEAN     = Re * MU_L / (RHO_L * D);
constexpr Float SIGMA      = RHO_L * Igor::sqr(U_MEAN) * D / We;
constexpr Float RHO_G      = RHO_L / rhor;
constexpr Float MU_G       = MU_L / mur;

constexpr Float Ca         = MU_L * U_MEAN / SIGMA;
constexpr Float La         = SIGMA * RHO_L * D / Igor::sqr(MU_L);

constexpr Float T_END      = L / (2.0 * 1.5 * U_MEAN);
constexpr Float DT_WRITE   = T_END / 100.0;
constexpr Float DT_MAX     = DT_WRITE;
constexpr Float CFL_MAX    = 0.9;

constexpr Index LEVEL      = 6;
constexpr Index NY         = 1 << LEVEL;
constexpr Index NX         = 1 << LEVEL;
constexpr Index NGHOST     = 1;

constexpr Float X_MIN      = 0.0;
constexpr Float X_MAX      = L;
constexpr Float Y_MIN      = 0.0;
constexpr Float Y_MAX      = L;

constexpr Circle<Float> C0 = {.x = 3.0 * L / 8.0, .y = L / 2.0, .r = D / 2.0};
constexpr auto vof0        = [](Float x, Float y) {
  return static_cast<Float>(C0.contains({.x = x, .y = y}));
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, Float /*t*/) -> Float {
  return -6.0 * U_MEAN / Igor::sqr(L) * y * (y - L);
}

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};

// = Config ========================================================================================

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
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  Igor::Info("Write result into `{}`", OUTPUT_DIR);
  std::putchar('\n');
  Igor::Info("Re       = {:g}", Re);
  Igor::Info("We       = {:g}", We);
  Igor::Info("Ca       = {:g}", Ca);
  Igor::Info("La       = {:g}", La);
  Igor::Info("rhor     = {:g}", rhor);
  Igor::Info("mur      = {:g}", mur);
  std::putchar('\n');
  Igor::Info("L        = {:g}", L);
  Igor::Info("D        = {:g}", D);
  Igor::Info("rho_l    = {:g}", RHO_L);
  Igor::Info("mu_l     = {:g}", MU_L);
  std::putchar('\n');
  Igor::Info("U_mean   = {:g}", U_MEAN);
  Igor::Info("sigma    = {:g}", SIGMA);
  Igor::Info("rho_g    = {:g}", RHO_G);
  Igor::Info("mu_g     = {:g}", MU_G);
  Igor::Info("tend     = {:g}", T_END);
  Igor::Info("dt_write = {:g}", DT_WRITE);

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas    = MU_G,
      .visc_liquid = MU_L,
      .rho_gas     = RHO_G,
      .rho_liquid  = RHO_L,
      .sigma       = SIGMA,
  };
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  VOF<Float, NX, NY, NGHOST> vof{};

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};
  Field2D<Float, NX, NY, NGHOST> rhoi{};

  Field2D<Float, NX + 1, NY, NGHOST> drho_u_stagdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drho_v_stagdt{};
  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};

  Field2D<Float, NX, NY, NGHOST> delta_p{};
  Field2D<Float, NX + 1, NY, NGHOST> delta_p_jump_u_stag{};
  Field2D<Float, NX, NY + 1, NGHOST> delta_p_jump_v_stag{};

  // Observation variables
  Float t             = 0.0;
  Float dt            = DT_MAX;

  Float U_max         = 0.0;
  Float V_max         = 0.0;

  Float div_max       = 0.0;

  Float vof_min       = 0.0;
  Float vof_max       = 0.0;
  Float vof_vol_error = 0.0;

  Float p_res         = 0.0;
  Index p_iter        = 0;
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

  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  // monitor.add_variable(&vof_integral, "int(vof)");
  // monitor.add_variable(&vof_loss, "loss(vof)");
  monitor.add_variable(&vof_vol_error, "max(vol. error)");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf(i, j) = quadrature(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  localize_cells(fs.x, fs.y, vof.ir);
  reconstruct_interface(fs, vof.vf, vof.ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U,
                             [&](Index i, Index j) { fs.curr.U(i, j) = U_in(fs.ym(j), 0.0); });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  calc_rho(vof.vf, fs);
  calc_visc(vof.vf, fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  vof_min = min(vof.vf);
  vof_max = max(vof.vf);
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
    calc_rho(vof.vf_old, fs);
    save_old_density(fs.curr, fs.old);

    advect_cells(fs, Ui, Vi, dt, vof, &vof_vol_error);
    // apply_neumann_bconds(vof.vf);
    calc_visc(vof.vf, fs);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update the density field to make the update consistent ==================================
      calc_drhodt(fs, drho_u_stagdt, drho_v_stagdt);
      update_density(drho_u_stagdt, drho_v_stagdt, dt, fs);
      apply_neumann_bconds(fs.curr.rho_u_stag);
      apply_neumann_bconds(fs.curr.rho_v_stag);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      update_velocity(drhoUdt, drhoVdt, dt, fs);
      // Boundary conditions
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
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    vof_min = min(vof.vf);
    vof_max = max(vof.vf);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
