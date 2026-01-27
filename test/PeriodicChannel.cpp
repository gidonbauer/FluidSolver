#include <cstddef>

#include <omp.h>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Utility.hpp"

#include "Common.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 5 * 43;
constexpr Index NY              = 43;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 5.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 60.0;
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

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Periodic{},
    .right  = Periodic{},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
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
auto main() -> int {
  omp_set_num_threads(4);

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory("test/output");
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

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
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("PeriodicChannel");
  bool any_test_failed = false;
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
        if (std::isnan(drhoUdt(i, j))) { Igor::Panic("NaN value in drhoUdt[{}, {}]", i, j); }
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        if (std::isnan(drhoVdt(i, j))) { Igor::Panic("NaN value in drhoVdt[{}, {}]", i, j); }
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

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

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      Index local_p_iter = 0;
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      {
        if (std::any_of(delta_p.get_data(), delta_p.get_data() + delta_p.size(), [](Float x) {
              return std::isnan(x);
            })) {
          Igor::Warn("Encountered NaN value in pressure correction.");
          return 1;
        }
      }

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

    {
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      if (std::abs(inflow - TOTAL_FLOW) > 1e-8) {
        Igor::Warn(
            "Inflow is not equal to TOTAL_FLOW at t={:.6e}: inflow={:.6e}, TOTAL_FLOW={:.6e}, "
            "error={:.6e}",
            t,
            inflow,
            TOTAL_FLOW,
            std::abs(inflow - TOTAL_FLOW));
        any_test_failed = true;
      }
      if (std::abs(mass_error) > 1e-8) {
        Igor::Warn("Outflow is not equal to inflow at t={:.6e}: inflow={:.6e}, outflow={:.6e}, "
                   "error={:.6e}",
                   t,
                   inflow,
                   outflow,
                   std::abs(outflow - inflow));
        any_test_failed = true;
      }
    }

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = abs_max(fs.curr.U);
    V_max   = abs_max(fs.curr.V);
    div_max = abs_max(div);
    p_max   = abs_max(fs.p);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  // = Perform tests ===============================================================================
  // - Test pressure ---------
  {
    constexpr Float TOL = 1e-4;
    for (Index i = 0; i < NX; ++i) {
      const auto ref_pressure = fs.p(i, 0);
      bool constant_pressure  = true;
      for (Index j = 0; j < fs.p.extent(1); ++j) {
        if (std::abs(fs.p(i, j) - ref_pressure) > TOL) { constant_pressure = false; }
      }
      if (!constant_pressure) {
        Igor::Warn("Non constant pressure along y-axis for x={}.", fs.xm(i));
        any_test_failed = true;
      }
    }

    const auto ref_dpdx = (fs.p(NX / 2 + 1, NY / 2) - fs.p(NX / 2, NY / 2)) / fs.dx;
    for (Index i = 1; i < fs.p.extent(0); ++i) {
      const auto dpdx = (fs.p(i, NY / 2) - fs.p(i - 1, NY / 2)) / fs.dx;
      if (std::abs(ref_dpdx - dpdx) > TOL) {
        Igor::Warn(
            "Non constant dpdx after x=60: Reference dpdx(x={:.6e})={:.6e}, dpdx(x={:.6e})={:.6e} "
            "=> error = {:.6e}",
            fs.x(NX / 2 + 1),
            ref_dpdx,
            fs.x(i),
            dpdx,
            std::abs(ref_dpdx - dpdx));
        any_test_failed = true;
      }
    }
  }

  // - Test U profile --------
  {
    constexpr Float TOL = 1e-3;
    auto u_analytical   = [&](Float y, Float dpdx) -> Float {
      return dpdx / (2 * VISC) * (y * y - y);
    };
    Field1D<Float, NY + 2 * NGHOST, 0> diff{};

    static_assert(X_MIN == 0.0, "Expected X_MIN to be 0 to make things a bit easier.");
    for_each_i(fs.x, [&](Index i) {
      for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
        const auto dpdx  = (fs.p(i, j) - fs.p(i - 1, j)) / fs.dx;
        diff(j + NGHOST) = std::abs(fs.curr.U(i, j) - u_analytical(fs.ym(j), dpdx));
      }
      const auto L1_error = simpsons_rule_1d(diff, Y_MIN, Y_MAX);
      if (L1_error > TOL) {
        Igor::Warn("U-velocity profile at x={} does not align with analytical solution: L1-error "
                   "is {:.6e}",
                   fs.x(i),
                   L1_error);
        any_test_failed = true;
      }
    });
  }

  // - Test V profile --------
  {
    constexpr Float TOL = 1e-7;
    for_each_i(fs.curr.V, [&](Index i, Index j) {
      if (std::abs(fs.curr.V(i, j)) > TOL) {
        Igor::Warn("V-velocity at ({:.6e}, {:.6e}) is not zero: {:.6e}",
                   fs.xm(i),
                   fs.y(j),
                   fs.curr.V(i, j));
        any_test_failed = true;
      }
    });
  }

  return any_test_failed ? 1 : 0;
}
