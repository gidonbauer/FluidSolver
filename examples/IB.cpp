#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                  = double;

constexpr Index NX           = 5 * 64;
constexpr Index NY           = 64;

constexpr Float X_MIN        = 0.0;
constexpr Float X_MAX        = 5.0;
constexpr Float Y_MIN        = 0.0;
constexpr Float Y_MAX        = 1.0;

constexpr Float T_END        = 4.0;
constexpr Float DT_MAX       = 1e-2;
constexpr Float CFL_MAX      = 0.5;
constexpr Float DT_WRITE     = 1e-2;

constexpr Float U_BCOND      = 5.0;
constexpr Float U_0          = 0.0;
constexpr Float VISC         = 1e-3;
constexpr Float RHO          = 1.0;

constexpr Float CX           = 1.0;
constexpr Float CY           = 0.5;
constexpr Float R0           = 0.1;
constexpr auto immersed_wall = [](Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_BCOND, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "output/IB/";
// = Config ========================================================================================

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
void calc_conserved_quantities_ib(const FS<Float, NX, NY>& fs,
                                  const Matrix<Float, NX + 1, NY>& ib_u_stag,
                                  const Matrix<Float, NX, NY + 1>& ib_v_stag,
                                  Float& mass,
                                  Float& momentum_x,
                                  Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      mass += (ib_u_stag[i, j] * fs.curr.rho_u_stag[i, j] +
               ib_u_stag[i + 1, j] * fs.curr.rho_u_stag[i + 1, j] +
               ib_v_stag[i, j] * fs.curr.rho_v_stag[i, j] +
               ib_v_stag[i, j + 1] * fs.curr.rho_v_stag[i, j + 1]) /
              4.0 * fs.dx * fs.dy;

      momentum_x += (ib_u_stag[i, j] * fs.curr.rho_u_stag[i, j] * fs.curr.U[i, j] +
                     ib_u_stag[i + 1, j] * fs.curr.rho_u_stag[i + 1, j] * fs.curr.U[i + 1, j]) /
                    2.0 * fs.dx * fs.dy;
      momentum_y += (ib_v_stag[i, j] * fs.curr.rho_v_stag[i, j] * fs.curr.V[i, j] +
                     ib_v_stag[i, j + 1] * fs.curr.rho_v_stag[i, j + 1] * fs.curr.V[i, j + 1]) /
                    2.0 * fs.dx * fs.dy;
    }
  }
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho_and_visc(fs);
  PS<Float, NX, NY> ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

  Matrix<Float, NX, NY> ib{};
  Matrix<Float, NX + 1, NY> ib_u_stag{};
  Matrix<Float, NX, NY + 1> ib_v_stag{};

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

  // Float p_max         = 0.0;
  Float p_res  = 0.0;
  Index p_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  vtk_writer.add_scalar("Immersed-wall", &ib);

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

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for (Index i = 0; i < ib_u_stag.extent(0); ++i) {
    for (Index j = 0; j < ib_u_stag.extent(1); ++j) {
      ib_u_stag[i, j] =
          quadrature(
              immersed_wall, fs.x[i] - fs.dx / 2.0, fs.x[i] + fs.dx / 2.0, fs.y[j], fs.y[j + 1]) /
          (fs.dx * fs.dy);
    }
  }

  for (Index i = 0; i < ib_v_stag.extent(0); ++i) {
    for (Index j = 0; j < ib_v_stag.extent(1); ++j) {
      ib_v_stag[i, j] =
          quadrature(
              immersed_wall, fs.x[i], fs.x[i + 1], fs.y[j] - fs.dy / 2.0, fs.y[j] + fs.dy / 2.0) /
          (fs.dx * fs.dy);
    }
  }
  interpolate_UV_staggered_field(ib_u_stag, ib_v_stag, ib);
  // = Initialize immersed boundaries ==============================================================

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

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  calc_conserved_quantities_ib(fs, ib_u_stag, ib_v_stag, mass, mom_x, mom_y);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
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
      for (Index i = 1; i < fs.curr.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.U.extent(1) - 1; ++j) {
          fs.curr.U[i, j] = (1.0 - ib_u_stag[i, j]) *
                            (fs.old.rho_u_stag[i, j] * fs.old.U[i, j] + dt * drhoUdt[i, j]) /
                            fs.curr.rho_u_stag[i, j];
        }
      }
      for (Index i = 1; i < fs.curr.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.V.extent(1) - 1; ++j) {
          fs.curr.V[i, j] = (1.0 - ib_v_stag[i, j]) *
                            (fs.old.rho_v_stag[i, j] * fs.old.V[i, j] + dt * drhoVdt[i, j]) /
                            fs.curr.rho_v_stag[i, j];
        }
      }

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      // Correct the outflow
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for (Index j = 0; j < NY; ++j) {
        fs.curr.U[NX, j] -= mass_error / (fs.curr.rho_u_stag[NX, j] * static_cast<Float>(NY));
      }

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      // = Apply pressure correction ===============================================================
      Index local_p_iter = 0;
      ps.setup(fs);
      ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      // Correct pressure
      for (Index i = 0; i < fs.p.extent(0); ++i) {
        for (Index j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      // Correct velocity
      for (Index i = 1; i < fs.curr.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.U.extent(1) - 1; ++j) {
          const auto dpdx  = (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx;
          const auto rho   = fs.curr.rho_u_stag[i, j];
          fs.curr.U[i, j] -= dpdx * dt / rho;
        }
      }
      for (Index i = 1; i < fs.curr.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.V.extent(1) - 1; ++j) {
          const auto dpdy  = (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy;
          const auto rho   = fs.curr.rho_v_stag[i, j];
          fs.curr.V[i, j] -= dpdy * dt / rho;
        }
      }
      // = Apply pressure correction ===============================================================
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    calc_conserved_quantities_ib(fs, ib_u_stag, ib_v_stag, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
