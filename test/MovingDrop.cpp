#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

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
using Float              = double;

constexpr Index NX       = 256;
constexpr Index NY       = 256;

constexpr Float X_MIN    = 0.0;
constexpr Float X_MAX    = 1.0;
constexpr Float Y_MIN    = 0.0;
constexpr Float Y_MAX    = 1.0;

constexpr Float T_END    = 0.5;
constexpr Float DT_MAX   = 1e-2;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = 1e-2;

constexpr Float U_DROP   = 1.0;
constexpr Float VISC_G   = 1e-3;  // 1e-0;
constexpr Float RHO_G    = 1.0;
constexpr Float VISC_L   = 1e-3;
constexpr Float RHO_L    = 1e9;

// constexpr Float SURFACE_TENSION = 1.0 / 20.0;  // sigma
constexpr Float CX  = 0.25;
constexpr Float CY  = 0.5;
constexpr Float R0  = 0.1;
constexpr auto vof0 = [](Float x, Float y) {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// TODO: Test case moving drop through stationary flow field

constexpr FlowBConds<Float> bconds{
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::NEUMANN, BCond::NEUMANN, BCond::NEUMANN, BCond::NEUMANN},
    .U     = {0.0, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

constexpr auto OUTPUT_DIR = "test/output/MovingDrop/";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_vof_stats(const FS<Float, NX, NY>& fs,
                    const Matrix<Float, NX, NY>& vof,
                    const Float init_vof_integral,
                    Float& min,
                    Float& max,
                    Float& integral,
                    Float& loss) noexcept {
  const auto [min_it, max_it] = std::minmax_element(vof.get_data(), vof.get_data() + vof.size());

  min                         = *min_it;
  max                         = *max_it;
  integral                    = integrate(fs.dx, fs.dy, vof);
  loss                        = init_vof_integral - integral;
}

// -------------------------------------------------------------------------------------------------
auto calc_center_of_mass(Float dx,
                         Float dy,
                         const Vector<Float, NX>& xm,
                         const Vector<Float, NY>& ym,
                         const Matrix<Float, NX, NY>& vof) -> std::array<Float, 2> {
  const auto vol   = integrate(dx, dy, vof);

  Float weighted_x = 0.0;
  Float weighted_y = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      weighted_x += xm[i] * vof[i, j];
      weighted_y += ym[j] * vof[i, j];
    }
  }
  weighted_x *= dx * dy;
  weighted_y *= dx * dy;

  return {weighted_x / vol, weighted_y / vol};
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{
      .visc_gas = VISC_G, .visc_liquid = VISC_L, .rho_gas = RHO_G, .rho_liquid = RHO_L};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  InterfaceReconstruction<NX, NY> ir{};
  Matrix<Float, NX, NY> vof_old{};
  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_smooth{};
  Matrix<Float, NX, NY> curv{};

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};
  Matrix<Float, NX, NY> rhoi{};

  Matrix<Float, NX + 1, NY> drho_u_stagdt{};
  Matrix<Float, NX, NY + 1> drho_v_stagdt{};
  Matrix<Float, NX + 1, NY> drhoUdt{};
  Matrix<Float, NX, NY + 1> drhoVdt{};
  Matrix<Float, NX, NY> delta_p{};

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

  Float vof_min       = 0.0;
  Float vof_max       = 0.0;
  Float vof_integral  = 0.0;
  Float vof_loss      = 0.0;
  Float vof_vol_error = 0.0;

  // Float p_max         = 0.0;
  Float p_res  = 0.0;
  Index p_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("density", &rhoi);
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
  // monitor.add_variable(&div_L1, "L1(div)");

  // monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  // monitor.add_variable(&vof_integral, "int(vof)");
  monitor.add_variable(&vof_loss, "loss(vof)");
  // monitor.add_variable(&vof_vol_error, "max(vol. error)");

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize VOF field ========================================================================
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      vof[i, j] = quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx * fs.dy);
    }
  }
  const Float init_vof_integral = integrate(fs.dx, fs.dy, vof);
  localize_cells(fs.x, fs.y, ir);
  reconstruct_interface(fs.x, fs.y, vof, ir);
  // = Initialize VOF field ========================================================================

  // = Initialize flow field =======================================================================
  for (Index i = 1; i < fs.curr.U.extent(0) - 1; ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      const auto U_minus = U_DROP * vof[i - 1, j];
      const auto U_plus  = U_DROP * vof[i, j];
      fs.curr.U[i, j]    = (U_minus + U_plus) / 2.0;
    }
  }

  apply_velocity_bconds(fs, bconds);

  calc_rho_and_visc(vof, fs);
  PS<Float, NX, NY> ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER);

  interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
  // p_max = max(fs.p);
  calc_vof_stats(fs, vof, init_vof_integral, vof_min, vof_max, vof_integral, vof_loss);
  calc_conserved_quantities(fs, mass, mom_x, mom_y);
  if (!vtk_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("MovingDrop");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);
    std::copy_n(vof.get_data(), vof.size(), vof_old.get_data());

    // = Update VOF field ==========================================================================
    reconstruct_interface(fs.x, fs.y, vof_old, ir);
    // TODO: Calculate viscosity from new VOF field
    calc_rho_and_visc(vof_old, fs);
    save_old_density(fs.curr, fs.old);

    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    advect_cells(fs, vof_old, Ui, Vi, dt, ir, vof, &vof_vol_error);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update the density field to make the update consistent ==================================
      calc_drhodt(fs, drho_u_stagdt, drho_v_stagdt);
      IGOR_ASSERT(std::none_of(drho_u_stagdt.get_data(),
                               drho_u_stagdt.get_data() + drho_u_stagdt.size(),
                               [](Float x) { return std::isnan(x); }),
                  "NaN value in drho_u_stagdt.");
      IGOR_ASSERT(std::none_of(drho_v_stagdt.get_data(),
                               drho_v_stagdt.get_data() + drho_v_stagdt.size(),
                               [](Float x) { return std::isnan(x); }),
                  "NaN value in drho_v_stagdt.");

      for (Index i = 1; i < fs.curr.rho_u_stag.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.rho_u_stag.extent(1) - 1; ++j) {
          fs.curr.rho_u_stag[i, j] = fs.old.rho_u_stag[i, j] + dt * drho_u_stagdt[i, j];
        }
      }
      apply_neumann_bconds(fs.curr.rho_u_stag);
      for (Index i = 1; i < fs.curr.rho_v_stag.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.rho_v_stag.extent(1) - 1; ++j) {
          fs.curr.rho_v_stag[i, j] = fs.old.rho_v_stag[i, j] + dt * drho_v_stagdt[i, j];
        }
      }
      apply_neumann_bconds(fs.curr.rho_v_stag);
      ps.setup(fs);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for (Index i = 1; i < fs.curr.U.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.U.extent(1) - 1; ++j) {
          fs.curr.U[i, j] = (fs.old.rho_u_stag[i, j] * fs.old.U[i, j] + dt * drhoUdt[i, j]) /
                            fs.curr.rho_u_stag[i, j];
        }
      }
      for (Index i = 1; i < fs.curr.V.extent(0) - 1; ++i) {
        for (Index j = 1; j < fs.curr.V.extent(1) - 1; ++j) {
          fs.curr.V[i, j] = (fs.old.rho_v_stag[i, j] * fs.old.V[i, j] + dt * drhoVdt[i, j]) /
                            fs.curr.rho_v_stag[i, j];
        }
      }

      // Boundary conditions
      apply_velocity_bconds(fs, bconds);

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      // TODO: Add capillary forces here.
#if 0
      smooth_vof_field(fs.xm, fs.ym, vof_old, vof_smooth);
      calc_curvature(fs.dx, fs.dy, vof_old, vof_smooth, curv);

      for (Index i = 0; i < NX; ++i) {
        for (Index j = 0; j < NY; ++j) {
          if (has_interface(vof_old, i, j)) {
            const auto dkappadx = (-curv[i + 1, j] - -curv[i - 1, j]) / (2.0 * fs.dx);
            const auto dkappady = (-curv[i, j + 1] - -curv[i, j - 1]) / (2.0 * fs.dy);

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
#endif

      Index local_p_iter = 0;
      if (!ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter)) {
        Igor::Warn("Pressure correction failed at t={}.", t);
      }
      p_iter += local_p_iter;
      {
        if (std::isnan(p_res) || std::any_of(delta_p.get_data(),
                                             delta_p.get_data() + delta_p.size(),
                                             [](Float x) { return std::isnan(x); })) {
          Igor::Warn("t={}, subiter={}: NaN value in pressure correction.", t, sub_iter);
          return 1;
        }
      }

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
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    interpolate_UV_staggered_field(fs.curr.rho_u_stag, fs.curr.rho_v_stag, rhoi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    // div_L1  = L1_norm(fs.dx, fs.dy, div) / ((X_MAX - X_MIN) * (Y_MAX - Y_MIN));
    // p_max = max(fs.p);
    calc_vof_stats(fs, vof, init_vof_integral, vof_min, vof_max, vof_integral, vof_loss);
    calc_conserved_quantities(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!vtk_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  const auto [cx, cy]    = calc_center_of_mass(fs.dx, fs.dy, fs.xm, fs.ym, vof);
  const auto cx_expected = CX + T_END * U_DROP;
  const auto cy_expected = CY;

  const auto error       = std::sqrt(Igor::sqr(cx - cx_expected) + Igor::sqr(cy - cy_expected));

  if (error > 1e-2) {
    Igor::Warn("Expected center of fluid = ({:.6e}, {:.6e})", cx_expected, cy_expected);
    Igor::Warn("Final center of fluid    = ({:.6e}, {:.6e})", cx, cy);
    Igor::Warn("Distance = {:.6e}", error);
    return 1;
  }
}
