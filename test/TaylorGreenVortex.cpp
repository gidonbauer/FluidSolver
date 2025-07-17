#include <cstddef>
#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 128;
constexpr Index NY              = 128;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 2.0 * std::numbers::pi_v<Float>;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 2.0 * std::numbers::pi_v<Float>;

constexpr Float T_END           = 5.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-2;

constexpr Float VISC            = 1e-1;
constexpr Float RHO             = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 2;

constexpr auto OUTPUT_DIR       = "test/output/TaylorGreenVortex/";
// = Config ========================================================================================

auto F(Float t) -> Float { return std::exp(-2.0 * VISC / RHO * t); }
auto u_analytical(Float x, Float y, Float t) -> Float { return std::sin(x) * std::cos(y) * F(t); }
auto v_analytical(Float x, Float y, Float t) -> Float { return -std::cos(x) * std::sin(y) * F(t); }

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  calc_rho_and_visc(fs);

  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float U_max   = 0.0;
  Float V_max   = 0.0;
  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0.0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");
  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  // = Initialize grid =============================================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * dx;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }
  init_mid_and_delta(fs);
  PS<Float, NX, NY> ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);
  // = Initialize grid =============================================================================

  // = Initialize flow field =======================================================================
  std::fill_n(fs.p.get_data(), fs.p.size(), 0.0);

  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      fs.curr.U[i, j] = u_analytical(fs.x[i], fs.ym[j], 0.0);
    }
  }
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      fs.curr.V[i, j] = v_analytical(fs.xm[i], fs.y[j], 0.0);
    }
  }

  // apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("TaylorGreenVortex");
  bool failed = false;
  while (t < T_END && !failed) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_state(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      // TODO: Handle density and interfaces
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
        for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.curr.U[i, j] = fs.old.U[i, j] + dt * drhoUdt[i, j] / RHO;
        }
      }
      for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
        for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
          // TODO: Need to interpolate rho for U- and V-staggered mesh
          fs.curr.V[i, j] = fs.old.V[i, j] + dt * drhoVdt[i, j] / RHO;
        }
      }

      // Boundary conditions
      // apply_velocity_bconds(fs, bconds);
      // Use custom Dirichlet boundary conditions
      {
        for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
          // LEFT
          fs.curr.U[0, j] = u_analytical(fs.x[0], fs.ym[j], t);
          // RIGHT
          fs.curr.U[fs.curr.U.extent(0) - 1, j] =
              u_analytical(fs.x[fs.curr.U.extent(0) - 1], fs.ym[j], t);
        }

        for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
          // BOTTOM
          fs.curr.U[i, 0] = u_analytical(fs.x[i], fs.ym[0], t);
          // TOP
          fs.curr.U[i, fs.curr.U.extent(1) - 1] =
              u_analytical(fs.x[i], fs.ym[fs.curr.U.extent(1) - 1], t);
        }

        for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
          // LEFT
          fs.curr.V[0, j] = v_analytical(fs.xm[0], fs.y[j], t);
          // RIGHT
          fs.curr.V[fs.curr.V.extent(0) - 1, j] =
              v_analytical(fs.xm[fs.curr.V.extent(0) - 1], fs.y[j], t);
        }

        for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
          // BOTTOM
          fs.curr.V[i, 0] = v_analytical(fs.xm[i], fs.y[0], t);
          // TOP
          fs.curr.V[i, fs.curr.V.extent(1) - 1] =
              v_analytical(fs.xm[i], fs.y[fs.curr.V.extent(1) - 1], t);
        }
      }

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      // TODO: Add capillary forces here.
      Index local_p_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
      }
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      for (Index i = 0; i < fs.p.extent(0); ++i) {
        for (Index j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      for (Index i = 1; i < fs.curr.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.U.extent(1) - 1; ++j) {
          fs.curr.U[i, j] -= (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / RHO;
        }
      }
      for (Index i = 1; i < fs.curr.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.V.extent(1) - 1; ++j) {
          fs.curr.V[i, j] -= (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / RHO;
        }
      }
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  if (failed) {
    Igor::Warn("TaylorGreenVortex failed.");
    return 1;
  }

  // = Perform tests ===============================================================================
  bool any_test_failed = false;
  const Float TOL      = 3.0 * Igor::sqr(std::max(fs.dx[0], fs.dy[0]));

  // TODO: Assumes equidistant spacing in x- and y-direction
  const auto vol = fs.dx[0] * fs.dy[0];

  // Test U
  Float L1_error_U = 0.0;
  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      L1_error_U += std::abs(fs.curr.U[i, j] - u_analytical(fs.x[i], fs.ym[j], T_END)) * vol;
    }
  }
  if (L1_error_U > TOL) {
    Igor::Warn("U profile is incorrect: L1 error = {}", L1_error_U);
    any_test_failed = true;
  }

  // Test V
  Float L1_error_V = 0.0;
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      L1_error_V += std::abs(fs.curr.V[i, j] - v_analytical(fs.xm[i], fs.y[j], T_END)) * vol;
    }
  }
  if (L1_error_V > TOL) {
    Igor::Warn("V profile is incorrect: L1 error = {}", L1_error_V);
    any_test_failed = true;
  }

  return any_test_failed ? 1 : 0;
}
