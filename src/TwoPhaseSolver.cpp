#include <cstddef>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

// #define FS_HYPRE_VERBOSE

#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float = double;

constexpr Index NX = 5 * 128;
constexpr Index NY = 128;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 5.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr Float T_END    = 0.5;  // 5.0;
constexpr Float DT_MAX   = 1e-4;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = 1e-2;

constexpr Float U_BCOND = 1.0;
constexpr Float U_0     = 0.0;
constexpr Float VISC_G  = 1e-3;
constexpr Float RHO_G   = 1.0;
constexpr Float VISC_L  = 1e-1;
constexpr Float RHO_L   = 10.0;

constexpr Float SURFACE_TENSION = 1.0 / 20.0;  // sigma
constexpr Float CX              = 1.0;
constexpr Float CY              = 0.5;
constexpr Float R0              = 0.25;
constexpr auto vof0             = [](Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};

constexpr int PRESSURE_MAX_ITER = 10;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER = 2;  // 5;

// Channel flow
// constexpr FlowBConds<Float> bconds{
//     //        LEFT              RIGHT           BOTTOM            TOP
//     .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
//     .U     = {U_BCOND, 0.0, 0.0, 0.0},
//     .V     = {0.0, 0.0, 0.0, 0.0},
// };

// Couette flow
constexpr FlowBConds<Float> bconds{
    //        LEFT            RIGHT           BOTTOM            TOP
    .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {0.0, 0.0, 0.0, U_BCOND},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "output/TwoPhaseSolver/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_vof_stats(const FS<Float, NX, NY>& fs,
                    const Matrix<Float, NX, NY>& vof,
                    const Float init_vof_integral,
                    Float& min,
                    Float& max,
                    Float& integral,
                    Float& loss,
                    Float& loss_prct) noexcept {
  const auto [min_it, max_it] = std::minmax_element(vof.get_data(), vof.get_data() + vof.size());

  min       = *min_it;
  max       = *max_it;
  integral  = integrate(fs.dx, fs.dy, vof);
  loss      = init_vof_integral - integral;
  loss_prct = 100.0 * loss / init_vof_integral;
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{
      .visc_gas = VISC_G, .visc_liquid = VISC_L, .rho_gas = RHO_G, .rho_liquid = RHO_L};

  constexpr auto dx = (X_MAX - X_MIN) / static_cast<Float>(NX);
  constexpr auto dy = (Y_MAX - Y_MIN) / static_cast<Float>(NY);
  PS<Float, NX, NY> ps(dx, dy, PRESSURE_TOL, PRESSURE_MAX_ITER);

  InterfaceReconstruction<NX, NY> ir{};
  Matrix<Float, NX, NY> vof_old{};
  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_smooth{};
  Matrix<Float, NX, NY> curv{};

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

  // Observation variables
  Float t  = 0.0;
  Float dt = DT_MAX;

  Float U_max = 0.0;
  Float V_max = 0.0;

  Float div_max = 0.0;
  Float div_L1  = 0.0;

  Float vof_min       = 0.0;
  Float vof_max       = 0.0;
  Float vof_integral  = 0.0;
  Float vof_loss      = 0.0;
  Float vof_loss_prct = 0.0;
  Float vof_vol_error = 0.0;

  Float pressure_res  = 0.0;
  Index pressure_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("density", &fs.rho);
  vtk_writer.add_scalar("viscosity", &fs.visc);
  vtk_writer.add_scalar("pressure", &fs.p);
  vtk_writer.add_scalar("divergence", &div);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_vector("velocity", &Ui, &Vi);
  vtk_writer.add_scalar("curvature", &curv);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");
  monitor.add_variable(&div_L1, "L1(div)");

  monitor.add_variable(&pressure_res, "res(p)");
  monitor.add_variable(&pressure_iter, "iter(p)");

  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  monitor.add_variable(&vof_integral, "int(vof)");
  monitor.add_variable(&vof_loss, "loss(vof)");
  // monitor.add_variable(&vof_loss_prct, "loss(vof) [%]");
  monitor.add_variable(&vof_vol_error, "max(vol. error)");
  // = Output ======================================================================================

  // = Initialize grid =============================================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * dx;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * dy;
  }
  init_mid_and_delta(fs);
  // = Initialize grid =============================================================================

  // = Initialize VOF field ========================================================================
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      vof[i, j] =
          quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx[i] * fs.dy[j]);
    }
  }
  const Float init_vof_integral = integrate(fs.dx, fs.dy, vof);
  localize_cells(fs.x, fs.y, ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  std::fill_n(fs.p.get_data(), fs.p.size(), 0.0);

  for (Index i = 0; i < fs.U.extent(0); ++i) {
    for (Index j = 0; j < fs.U.extent(1); ++j) {
      // fs.U[i, j] = U_0;
      if (i == 0 || i == fs.U.extent(0) - 1) {
        fs.U[i, j] = fs.ym[j];
      } else {
        const auto v = (vof[i, j] + vof[i - 1, j]) / 2.0;
        fs.U[i, j]   = fs.ym[j] * (1.0 - v);
      }
    }
  }
  for (Index i = 0; i < fs.V.extent(0); ++i) {
    for (Index j = 0; j < fs.V.extent(1); ++j) {
      fs.V[i, j] = 0.0;
    }
  }

  apply_velocity_bconds(fs, bconds);

  calc_rho_and_visc(vof, fs);

  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);
  calc_divergence(fs, div);
  U_max   = max(fs.U);
  V_max   = max(fs.V);
  div_max = max(div);
  div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
  calc_vof_stats(
      fs, vof, init_vof_integral, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
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
    std::copy_n(fs.U.get_data(), fs.U.size(), fs.U_old.get_data());
    std::copy_n(fs.V.get_data(), fs.V.size(), fs.V_old.get_data());
    std::copy_n(vof.get_data(), vof.size(), vof_old.get_data());

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs.x, fs.y, vof_old, ir);

    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    advect_cells(fs, vof_old, Ui, Vi, dt, ir, vof, &vof_vol_error);
    calc_rho_and_visc(vof_old, fs);

    pressure_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.U, fs.U_old);
      calc_mid_time(fs.V, fs.V_old);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for (Index i = 1; i < fs.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.U.extent(1) - 1; ++j) {
          const auto rho = (fs.rho[i, j] + fs.rho[i - 1, j]) / 2.0;
          fs.U[i, j]     = fs.U_old[i, j] + dt * drhoUdt[i, j] / rho;
        }
      }
      for (Index i = 1; i < fs.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.V.extent(1) - 1; ++j) {
          const auto rho = (fs.rho[i, j] + fs.rho[i, j - 1]) / 2.0;
          fs.V[i, j]     = fs.V_old[i, j] + dt * drhoVdt[i, j] / rho;
        }
      }

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      calc_divergence(fs, div);
      // TODO: Add capillary forces here.
#ifdef ANALYTIC_CURVATURE
      {
        auto get_curvature = [](Float x, Float y) {
          return 1.0 / std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));
        };
        for (Index i = 0; i < NX; ++i) {
          for (Index j = 0; j < NY; ++j) {
            curv[i, j] = get_curvature(fs.xm[i], fs.ym[j]);
          }
        }
      }
#else
      smooth_vof_field(fs.xm, fs.ym, vof_old, vof_smooth);
      calc_curvature(dx, dy, vof_old, vof_smooth, curv);
#endif

      for (Index i = 0; i < NX; ++i) {
        for (Index j = 0; j < NY; ++j) {
          if (has_interface(vof_old, i, j)) {
            const auto dkappadx = (curv[i + 1, j] - curv[i - 1, j]) / (2.0 * fs.dx[i]);
            const auto dkappady = (curv[i, j + 1] - curv[i, j - 1]) / (2.0 * fs.dy[j]);

            IGOR_ASSERT((ir.interface[i, j].getNumberOfPlanes() == 1),
                        "Expected exactly one plane but got {}",
                        ir.interface[i, j].getNumberOfPlanes());

            const IRL::Normal n = ir.interface[i, j][0].normal();
            IGOR_ASSERT(std::abs(n[2]) < 1e-12,
                        "Expected z-component of normal to be 0 but is {:.6e}",
                        n[2]);

            div[i, j] += dt * 2.0 * SURFACE_TENSION * (n[0] * dkappadx + n[1] * dkappady);
          }
        }
      }

      Index local_pressure_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &pressure_res, &local_pressure_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
        failed = true;
      }
      pressure_iter += local_pressure_iter;

      shift_pressure_to_zero(fs, delta_p);
      for (Index i = 0; i < fs.p.extent(0); ++i) {
        for (Index j = 0; j < fs.p.extent(1); ++j) {
          fs.p[i, j] += delta_p[i, j];
        }
      }

      for (Index i = 1; i < fs.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.U.extent(1) - 1; ++j) {
          const auto rho = (fs.rho[i, j] + fs.rho[i - 1, j]) / 2.0;
          fs.U[i, j] -= (delta_p[i, j] - delta_p[i - 1, j]) / fs.dx[i] * dt / rho;
        }
      }
      for (Index i = 1; i < fs.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.V.extent(1) - 1; ++j) {
          const auto rho = (fs.rho[i, j] + fs.rho[i, j - 1]) / 2.0;
          fs.V[i, j] -= (delta_p[i, j] - delta_p[i, j - 1]) / fs.dy[j] * dt / rho;
        }
      }
    }

    t += dt;
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    U_max   = max(fs.U);
    V_max   = max(fs.V);
    div_max = max(div);
    div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
    calc_vof_stats(
        fs, vof, init_vof_integral, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
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
