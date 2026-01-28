#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Geometry.hpp"
#include "IB.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

#include "../test/Common.hpp"

// = Config ========================================================================================
using Float                    = double;

constexpr Float X_MIN          = 0.0;
constexpr Float X_MAX          = 5.0;
constexpr Float Y_MIN          = 0.0;
constexpr Float Y_MAX          = 5.0;
constexpr Float CHANNEL_HEIGHT = 1.0;
constexpr Float CHANNEL_LENGTH = X_MAX - X_MIN;
constexpr Float CHANNEL_OFFSET = 2.0;

#ifndef FS_SCALING_N
#error "Need to define `FS_SCALING_N`, the number of cells in x- and y-direction"
#endif  // FS_SCALING_N
constexpr Index NX         = (1 << FS_SCALING_N) + 1;
constexpr Index NY         = NX;
constexpr Index NGHOST     = 1;

constexpr Float T_END      = 10.0;
constexpr Float DT_MAX     = 1e-1;
constexpr Float CFL_MAX    = 0.25;
constexpr Float DT_WRITE   = T_END / 100.0;

constexpr Float VISC       = 1e-2;
constexpr Float RHO        = 1.0;
constexpr Float P0         = 0.2;

constexpr Rect wall_bottom = {
    .x = X_MIN - 1.0,
    .y = Y_MIN,
    .w = X_MAX - X_MIN + 1.0,
    .h = CHANNEL_OFFSET,
};
constexpr Rect wall_top = {
    .x = X_MIN - 1.0,
    .y = Y_MAX - CHANNEL_OFFSET,
    .w = X_MAX - X_MIN + 1.0,
    .h = CHANNEL_OFFSET,
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, [[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
  const Point p{.x = (X_MAX - X_MIN) / 2.0, .y = y};
  if (wall_bottom.contains(p) || wall_top.contains(p)) { return 0.0; }

  const Float y_off = y - CHANNEL_OFFSET;
  return P0 * CHANNEL_HEIGHT / (VISC * CHANNEL_LENGTH) * y_off * (1.0 - y_off / CHANNEL_HEIGHT);
}

constexpr Float U_AVG = quadrature<64>([](Float y) { return U_in(y, 0.0); },
                                       CHANNEL_OFFSET,
                                       CHANNEL_OFFSET + CHANNEL_HEIGHT) /
                        CHANNEL_HEIGHT;
constexpr Float Re = RHO * U_AVG * CHANNEL_HEIGHT / VISC;

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
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
auto main(int argc, char** argv) -> int {
  bool write_csv_header = false;
  bool write_csv        = false;
#define USAGE Igor::Error("Usage: {} [--write-csv-header] [--write-csv]", *argv)
  for (int i = 1; i < argc; ++i) {
    using namespace std::string_literals;
    if (argv[i] == "--write-csv-header"s) {
      write_csv_header = true;
    } else if (argv[i] == "--write-csv"s) {
      write_csv = true;
    } else {
      USAGE;
      Igor::Error("Unknown flag `{}`", argv[i]);
    }
  }
#undef USAGE

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory("scaling/output") + std::to_string(FS_SCALING_N);
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);
  ps.set_pressure_operator(fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  Field2D<Float, NX, NY, NGHOST> ib{};
  Field2D<Float, NX + 1, NY, NGHOST> ib_corr_u_stag{};
  Field2D<Float, NX, NY + 1, NGHOST> ib_corr_v_stag{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0;
  Float p_max   = 0.0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("Immersed-wall", &ib);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");

  monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for_each_a<Exec::Parallel>(ib, [&](Index i, Index j) {
    ib(i, j) = quadrature(
                   [](Float x, Float y) -> Float {
                     const Point p{.x = x, .y = y};
                     return static_cast<Float>(wall_bottom.contains(p) || wall_top.contains(p));
                   },
                   fs.x(i),
                   fs.x(i + 1),
                   fs.y(j),
                   fs.y(j + 1)) /
               (fs.dx * fs.dy);
  });
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  fill(fs.curr.U, 0.0);
  fill(fs.curr.V, 0.0);
  apply_velocity_bconds(fs, bconds, t);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  p_max   = max(fs.p);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      update_velocity(drhoUdt, drhoVdt, dt, fs);

      fill(ib_corr_u_stag, 0.0);
      fill(ib_corr_v_stag, 0.0);
      calc_ib_correction_shape(wall_bottom, fs.dx, fs.dy, fs.x, fs.ym, ib_corr_u_stag);
      calc_ib_correction_shape(wall_bottom, fs.dx, fs.dy, fs.xm, fs.y, ib_corr_v_stag);

      calc_ib_correction_shape(wall_top, fs.dx, fs.dy, fs.x, fs.ym, ib_corr_u_stag);
      calc_ib_correction_shape(wall_top, fs.dx, fs.dy, fs.xm, fs.y, ib_corr_v_stag);

      correct_velocity_ib_implicit(ib_corr_u_stag, ib_corr_v_stag, dt, fs);

      apply_velocity_bconds(fs, bconds, t);
      // = Update flow field =======================================================================

      // = Correct the outflow =====================================================================
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });
      // = Correct the outflow =====================================================================

      // = Apply pressure correction ===============================================================
      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_p_iter = 0;
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
      // = Apply pressure correction ===============================================================
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    p_max   = max(fs.p);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  // = Check results ===============================================================================
  const Float dpdx_avg = (fs.p(NX - 1, NY / 2) - fs.p(0, NY / 2)) / CHANNEL_LENGTH;
  const Float dpdx_exp = -2.0 * P0 / CHANNEL_LENGTH;
  Float dpdx_error     = 0.0;
  Float U_L1_error     = 0.0;
  {
    constexpr Index j_mid = NY / 2;
    for_each_i(fs.xm, [&](Index i) {
      const Float dpdx  = (fs.p(i + 1, j_mid) - fs.p(i - 1, j_mid)) / (2.0 * fs.dx);
      dpdx_error       += Igor::sqr(dpdx_exp - dpdx);
    });
    dpdx_error /= static_cast<Float>(NX);
  }

  {
    auto u_analytical = [=](Float y) -> Float {
      const Point p{.x = (X_MAX - X_MIN) / 2.0, .y = y};
      if (wall_bottom.contains(p) || wall_top.contains(p)) { return 0.0; }
      const auto y_off = y - CHANNEL_OFFSET;
      return dpdx_exp / (2 * VISC) * (y_off * y_off - y_off);
    };

    Field1D<Float, NY + 2 * NGHOST, 0> diff{};
    for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
      diff(j + NGHOST) = std::abs(fs.curr.U(NX / 2, j) - u_analytical(fs.ym(j)));
    }
    U_L1_error = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
  }

  if (write_csv_header) { std::cout << "NX,NY,T_END,Re,dpdx_avg,dpdx_exp,MSE_dpdx,L1_error_U\n"; }
  if (write_csv) {
    std::cout << NX << ',';
    std::cout << NY << ',';
    std::cout << std::scientific << std::setprecision(6) << T_END << ',';
    std::cout << std::scientific << std::setprecision(6) << Re << ',';
    std::cout << std::scientific << std::setprecision(6) << dpdx_avg << ',';
    std::cout << std::scientific << std::setprecision(6) << dpdx_exp << ',';
    std::cout << std::scientific << std::setprecision(6) << dpdx_error << ',';
    std::cout << std::scientific << std::setprecision(6) << U_L1_error << '\n';
  } else {
    Igor::Info("NX         = {}", NX);
    Igor::Info("NY         = {}", NY);
    Igor::Info("T_END      = {:.6e}", T_END);
    Igor::Info("Re         = {:.6e}", Re);
    Igor::Info("dpdx_avg   = {:.6e}", dpdx_avg);
    Igor::Info("dpdx_exp   = {:.6e}", dpdx_exp);
    Igor::Info("MSE dpdx   = {:.6e}", dpdx_error);
    Igor::Info("L1-error U = {:.6e}", U_L1_error);
  }
}
