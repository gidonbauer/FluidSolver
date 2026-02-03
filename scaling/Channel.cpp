#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>

#define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 5.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 20.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 1.0;

constexpr Float U_INIT          = 1.0;
constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 0.5;
constexpr Float TOTAL_FLOW      = (Y_MAX - Y_MIN) * U_INIT * RHO;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

#ifdef PERIODIC
constexpr FlowBConds<Float> bconds{
    .left   = Periodic{},
    .right  = Periodic{},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
#else
constexpr auto U_in(Float y, Float /*t*/) -> Float {
  constexpr Float DPDX = -12.0 * VISC * TOTAL_FLOW / RHO;
  return DPDX / (2.0 * VISC) * (y * y - y);
}

constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
#endif  // PERIODIC
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
template <Index NX, Index NY>
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0;
  outflow = 0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag(-NGHOST, j) * fs.curr.U(-NGHOST, j) * fs.dy;
    outflow += fs.curr.rho_u_stag(NX + NGHOST, j) * fs.curr.U(NX + NGHOST, j) * fs.dy;
  });
  mass_error = outflow - inflow;
}

// -------------------------------------------------------------------------------------------------
template <Index N>
auto run_simulation(bool print_csv_format) -> bool {
  constexpr Index NY = (1 << N) + 1;
  constexpr Index NX = 5LL * NY;

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory("scaling/output") + std::to_string(N) + "/";
  if (!init_output_directory(OUTPUT_DIR)) { return false; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  Float t          = 0.0;
  Float dt         = DT_MAX;

  Float mass       = 0.0;
  Float mom_x      = 0.0;
  Float mom_y      = 0.0;

  Float U_max      = 0.0;
  Float V_max      = 0.0;
  Float div_max    = 0.0;

  Float p_res      = 0.0;
  Index p_iter     = 0;
  Float p_max      = 0.0;

  Float inflow     = 0;
  Float outflow    = 0;
  Float mass_error = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");
  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");
  // monitor.add_variable(&inflow, "inflow");
  // monitor.add_variable(&outflow, "outflow");
  monitor.add_variable(&mass_error, "mass error");
  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");

  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  // = Initialize pressure solver ==================================================================
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);
  ps.set_pressure_operator(fs);
  // = Initialize pressure solver ==================================================================

  // = Initialize flow field =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = U_INIT; });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = abs_max(fs.curr.U);
  V_max   = abs_max(fs.curr.V);
  div_max = abs_max(div);
  p_max   = abs_max(fs.p);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return false; }
  monitor.write();
  // = Initialize flow field =======================================================================

  const auto t_begin = std::chrono::high_resolution_clock::now();
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_state(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

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

#ifdef PERIODIC
      // = Force total flow ========================================================================
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      const Float inflow_error  = TOTAL_FLOW - inflow;
      const Float outflow_error = TOTAL_FLOW - outflow;
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(-NGHOST, j)     += inflow_error / (fs.curr.rho_u_stag(-NGHOST, j) * fs.dy *
                                                 static_cast<Float>(NY + 2 * NGHOST));
        fs.curr.U(NX + NGHOST, j) += outflow_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * fs.dy *
                                                      static_cast<Float>(NY + 2 * NGHOST));
      });
      // = Force total flow ========================================================================
#else
      // = Correct outflow =========================================================================
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });
      // = Correct outflow =========================================================================
#endif  // PERIODIC

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      Index local_p_iter = 0;
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) -=
            (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx * dt / fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) -=
            (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy * dt / fs.curr.rho_v_stag(i, j);
      });
    }
    t += dt;

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    p_max   = abs_max(fs.p);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return false; }
    }
    monitor.write();
  }
  const auto t_dur =
      std::chrono::duration<Float>(std::chrono::high_resolution_clock::now() - t_begin);

  // = Perform tests ===============================================================================
  // - Test pressure ---------
  Float pressure_error = 0.0;
  for (Index i = 0; i < NX; ++i) {
    const auto ref_pressure = fs.p(i, 0);
    for (Index j = 0; j < fs.p.extent(1); ++j) {
      pressure_error += std::abs(fs.p(i, j) - ref_pressure);
    }
  }
  pressure_error      /= static_cast<Float>(NX * fs.p.extent(1));

  Float dpdx_error     = 0.0;
  const auto ref_dpdx  = (fs.p(NX / 2 + 1, NY / 2) - fs.p(NX / 2, NY / 2)) / fs.dx;
  for (Index i = 1; i < fs.p.extent(0); ++i) {
    const auto dpdx  = (fs.p(i, NY / 2) - fs.p(i - 1, NY / 2)) / fs.dx;
    dpdx_error      += std::abs(ref_dpdx - dpdx);
  }
  dpdx_error /= static_cast<Float>(fs.p.extent(0) - 1);

  // - Test U profile --------
  Float U_error     = 0.0;
  auto u_analytical = [&](Float y, Float dpdx) -> Float { return dpdx / (2 * VISC) * (y * y - y); };
  Field1D<Float, NY, 0> diff{};

  static_assert(X_MIN == 0.0, "Expected X_MIN to be 0 to make things a bit easier.");
#if 0
  for_each_i(fs.x, [&](Index i) {
    for_each_i(fs.ym, [&](Index j) {
      const auto dpdx = (fs.p(i, j) - fs.p(i - 1, j)) / fs.dx;
      diff(j)         = std::abs(fs.curr.U(i, j) - u_analytical(fs.ym(j), dpdx));
    });
    U_error += trapezoidal_rule(std::span(diff.get_data(), diff.size()),
                                std::span(&fs.ym(0), fs.ym.extent(0)));
  });
  U_error /= static_cast<Float>(fs.x.extent(0));
#else
  {
    const Index i = NX / 2;
    for_each_i(fs.ym, [&](Index j) {
      const auto dpdx = (fs.p(i, j) - fs.p(i - 1, j)) / fs.dx;
      diff(j)         = std::abs(fs.curr.U(i, j) - u_analytical(fs.ym(j), dpdx));
    });
    U_error = trapezoidal_rule(std::span(diff.get_data(), diff.size()),
                               std::span(&fs.ym(0), fs.ym.extent(0)));
  }
#endif

  // - Test V profile --------
  Float V_error = 0.0;
  for_each_i(fs.curr.V, [&](Index i, Index j) { V_error += std::abs(fs.curr.V(i, j)); });
  V_error /= static_cast<Float>(fs.curr.V.extent(0) * fs.curr.V.extent(1));

  if (print_csv_format) {
    // NX,NY,dx,dy,pressure_error,dpdx_error,U_error,V_error,runtime_s
    std::cout << Igor::detail::format("{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
                                      NX,
                                      NY,
                                      fs.dx,
                                      fs.dy,
                                      pressure_error,
                                      dpdx_error,
                                      U_error,
                                      V_error,
                                      t_dur.count());
  } else {
    Igor::Info("NX             = {}", NX);
    Igor::Info("NY             = {}", NY);
    Igor::Info("dx             = {:.6e}", fs.dx);
    Igor::Info("dy             = {:.6e}", fs.dy);
    Igor::Info("pressure_error = {:.6e}", pressure_error);
    Igor::Info("dpdx_error     = {:.6e}", dpdx_error);
    Igor::Info("U_error        = {:.6e}", U_error);
    Igor::Info("V_error        = {:.6e}", V_error);
    Igor::Info("Runtime [s]    = {:.6e}", t_dur.count());
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
auto main(int argc, char** argv) -> int {
  bool print_csv_format = false;
  for (int i = 1; i < argc; ++i) {
    using namespace std::string_literals;
    if (argv[i] == "--csv"s || argv[i] == "-csv"s) {
      print_csv_format = true;
    } else {
      Igor::Error("Usage: {} [--csv]", argv[0]);
      return 1;
    }
  }

  if (print_csv_format) {
    std::cout << "NX,NY,dx,dy,pressure_error,dpdx_error,U_error,V_error,runtime_s\n";
  }
  run_simulation<3>(print_csv_format);
  run_simulation<4>(print_csv_format);
  run_simulation<5>(print_csv_format);
  run_simulation<6>(print_csv_format);
  run_simulation<7>(print_csv_format);
  // run_simulation<8>(print_csv_format);
  // run_simulation<9>(print_csv_format);
  // run_simulation<10>(print_csv_format);
  return 0;
}
