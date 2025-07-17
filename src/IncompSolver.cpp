#include <cstddef>
#include <numeric>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

// #define FS_HYPRE_VERBOSE

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 640;
constexpr Index NY              = 64;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 10.0;  // 10.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 60.0;  // 10.0;
constexpr Float DT_MAX          = 1e-1;
constexpr Float CFL_MAX         = 0.9;
constexpr Float DT_WRITE        = 0.5;

constexpr Float U_BCOND         = 1.0;
constexpr Float U_0             = 0.0;
constexpr Float VISC            = 1e-3;  // 1e-1;
constexpr Float RHO             = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_BCOND, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

// // Couette flow
// constexpr FlowBConds<Float> bconds{
//     //        LEFT            RIGHT           BOTTOM            TOP
//     .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
//     .U     = {0.0, 0.0, 0.0, U_BCOND},
//     .V     = {0.0, 0.0, 0.0, 0.0},
// };

constexpr auto OUTPUT_DIR = "output/IncompSolver/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_velocity_stats(const Matrix<Float, NX + 1, NY>& U,
                         const Matrix<Float, NX, NY + 1>& V,
                         const Matrix<Float, NX, NY>& div,
                         Float& U_max,
                         Float& V_max,
                         Float& div_max) {
  U_max = std::transform_reduce(
      U.get_data(),
      U.get_data() + U.size(),
      Float{0.0},
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });

  V_max = std::transform_reduce(
      V.get_data(),
      V.get_data() + V.size(),
      Float{0.0},
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });

  div_max = std::transform_reduce(
      div.get_data(),
      div.get_data() + div.size(),
      Float{0.0},
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });
}

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0.0;
  outflow = 0.0;
  for (Index j = 0; j < NY; ++j) {
    inflow  += fs.curr.rho_u_stag[0, j] * fs.curr.U[0, j];
    outflow += fs.curr.rho_u_stag[NX, j] * fs.curr.U[NX, j];
  }
  mass_error = outflow - inflow;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

  Float t             = 0.0;
  Float dt            = DT_MAX;

  Float U_max         = 0.0;
  Float V_max         = 0.0;
  Float div_max       = 0.0;

  Float inflow        = 0.0;
  Float outflow       = 0.0;
  Float mass_error    = 0.0;

  Float pressure_res  = 0.0;
  Index pressure_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  calc_rho_and_visc(fs);

  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");
  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&pressure_res, "res(p)");
  monitor.add_variable(&pressure_iter, "iter(p)");
  monitor.add_variable(&inflow, "inflow");
  monitor.add_variable(&outflow, "outflow");
  monitor.add_variable(&mass_error, "mass error");
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
      fs.curr.U[i, j] = U_0;
    }
  }
  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      fs.curr.V[i, j] = 0.0;
    }
  }
  apply_velocity_bconds(fs, bconds);
  calc_inflow_outflow(fs, inflow, outflow, mass_error);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  calc_velocity_stats(fs.curr.U, fs.curr.V, div, U_max, V_max, div_max);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  bool failed = false;
  Igor::ProgressBar<Float> pbar(T_END, 67);
  while (t < T_END && !failed) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_state(fs.curr, fs.old);

    pressure_iter = 0;
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
      apply_velocity_bconds(fs, bconds);

      // Correct the outflow
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for (Index j = 0; j < NY; ++j) {
        fs.curr.U[NX, j] -= mass_error / (fs.curr.rho_u_stag[NX, j] * static_cast<Float>(NY - 2));
      }

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      // TODO: Add capillary forces here.
      Index local_pressure_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &pressure_res, &local_pressure_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
      }
      pressure_iter += local_pressure_iter;

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
    calc_inflow_outflow(fs, inflow, outflow, mass_error);

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    calc_velocity_stats(fs.curr.U, fs.curr.V, div, U_max, V_max, div_max);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
    pbar.update(dt);
  }
  std::cout << '\n';

  if (failed) {
    Igor::Warn("Solver did not finish successfully.");
    return 1;
  } else {
    Igor::Info("Solver finish successfully.");
  }
}
