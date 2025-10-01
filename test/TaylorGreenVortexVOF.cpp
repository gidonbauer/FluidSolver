#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float               = double;

constexpr Index NX        = 128;
constexpr Index NY        = 128;
constexpr Index NGHOST    = 1;

constexpr Float X_MIN     = 0.0;
constexpr Float X_MAX     = 2.0 * std::numbers::pi_v<Float>;
constexpr Float Y_MIN     = 0.0;
constexpr Float Y_MAX     = 2.0 * std::numbers::pi_v<Float>;
constexpr auto DX         = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY         = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float VISC      = 1e-1;
constexpr Float RHO       = 0.9;

Float INIT_VF_INT         = 0.0;  // NOLINT

constexpr Float DT_MAX    = 1e-2;
constexpr Float CFL_MAX   = 0.5;
constexpr Float T_END     = 5.0;
constexpr Float DT_WRITE  = 5e-2;

constexpr auto OUTPUT_DIR = "test/output/TaylorGreenVortexVOF";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void get_vof_stats(const Matrix<Float, NX, NY, NGHOST>& vf,
                   Float& min,
                   Float& max,
                   Float& integral,
                   Float& loss,
                   Float& loss_prct) noexcept {
  const auto [min_it, max_it] = std::minmax_element(vf.get_data(), vf.get_data() + vf.size());

  min                         = *min_it;
  max                         = *max_it;
  integral                    = integrate<true>(DX, DY, vf);
  loss                        = INIT_VF_INT - integral;
  loss_prct                   = 100.0 * loss / INIT_VF_INT;
}

// -------------------------------------------------------------------------------------------------
auto check_vof(Float vof_min, Float vof_max, Float vof_integral, Float max_volume_error) -> bool {
  if (std::abs(vof_min) > 1e-12) {
    Igor::Warn("Expected minimum VOF value to be 0 but is {:.6e}: error = {:.6e}",
               vof_min,
               std::abs(vof_min));
    return false;
  }

  if (std::abs(vof_max - 1.0) > 1e-12) {
    Igor::Warn("Expected maximum VOF value to be 1 but is {:.6e}: error = {:.6e}",
               vof_max,
               std::abs(vof_max - 1.0));
    return false;
  }

  if (std::abs(vof_integral - INIT_VF_INT) > 1e-10) {
    Igor::Warn("Expected integral of vof to be {:.6e} but is {:.6e}: error = {:.6e}",
               INIT_VF_INT,
               vof_integral,
               std::abs(vof_integral - INIT_VF_INT));
    return false;
  }
  if (max_volume_error > 1e-15) {
    Igor::Warn("Exceeded max. allowed volume error ({:.6e})", max_volume_error);
    return false;
  }
  return true;
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] constexpr auto F(Float t) -> Float { return std::exp(-2.0 * VISC / RHO * t); }
[[nodiscard]] constexpr auto u_analytical(Float x, Float y, Float t) -> Float {
  return std::sin(x) * std::cos(y) * F(t);
}
[[nodiscard]] constexpr auto v_analytical(Float x, Float y, Float t) -> Float {
  return -std::cos(x) * std::sin(y) * F(t);
}

void constexpr set_velocity(const Vector<Float, NX + 1, NGHOST>& x,
                            const Vector<Float, NY + 1, NGHOST>& y,
                            const Vector<Float, NX, NGHOST>& xm,
                            const Vector<Float, NY, NGHOST>& ym,
                            Float t,
                            Matrix<Float, NX + 1, NY, NGHOST>& U,
                            Matrix<Float, NX, NY + 1, NGHOST>& V) {
  for_each_a<Exec::Parallel>(U, [&](Index i, Index j) { U(i, j) = u_analytical(x(i), ym(j), t); });
  for_each_a<Exec::Parallel>(V, [&](Index i, Index j) { V(i, j) = v_analytical(xm(i), y(j), t); });
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};

  VOF<Float, NX, NY, NGHOST> vof{};

  VTKWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("VOF", &vof.vf);
  data_writer.add_scalar("div", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, vof.ir);
  // = Setup grid and cell localizers ==============================================================

  // = Setup velocity and vof field ================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    auto is_in = [](Float x, Float y) -> Float {
      return Igor::sqr(x - std::numbers::pi) + Igor::sqr(y - 1.5 * std::numbers::pi) <=
             Igor::sqr(0.5);
    };

    vof.vf(i, j) = quadrature(is_in, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  reconstruct_interface(fs, vof.vf, vof.ir);

  set_velocity(fs.x, fs.y, fs.xm, fs.ym, 0.0, fs.curr.U, fs.curr.V);
  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  calc_rho(fs);
  calc_visc(fs);
  Float max_div = abs_max(div);

  if (!data_writer.write(0.0)) { return 1; }
  INIT_VF_INT = integrate<true>(fs.dx, fs.dy, vof.vf);
  // = Setup velocity and vof field ================================================================

  Float t                = 0.0;
  Float dt               = DT_MAX;
  Float max_volume_error = 0.0;
  Float vof_min          = 0.0;
  Float vof_max          = 0.0;
  Float vof_integral     = 0.0;
  Float vof_loss         = 0.0;
  Float vof_loss_prct    = 0.0;

  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&max_volume_error, "max(vol. error)");
  monitor.add_variable(&vof_min, "min(vof)");
  monitor.add_variable(&vof_max, "max(vof)");
  monitor.add_variable(&vof_integral, "int(vof)");
  monitor.add_variable(&vof_loss, "loss(vof)");
  monitor.add_variable(&vof_loss_prct, "loss(vof) [%]");
  monitor.add_variable(&max_div, "max(div)");

  get_vof_stats(vof.vf, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
  if (!check_vof(vof_min, vof_max, vof_integral, max_volume_error)) { return 1; }

  Index counter = 0;
  Igor::ScopeTimer timer("TaylorGreenVortexVOF");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);
    std::copy_n(vof.vf.get_data(), vof.vf.size(), vof.vf_old.get_data());

    // = Reconstruct the interface =================================================================
    reconstruct_interface(fs, vof.vf, vof.ir);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, counter),
                          fs.x,
                          fs.y,
                          vof.ir.interface)) {
        return 1;
      }
    }

    // = Advect cells ==============================================================================
    advect_cells(fs, Ui, Vi, dt, vof, &max_volume_error);

    // = Update velocity field according to analytical solution ====================================
    t += dt;
    set_velocity(fs.x, fs.y, fs.xm, fs.ym, t, fs.curr.U, fs.curr.V);
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    max_div = abs_max(div);

    if (should_save(t, dt, DT_WRITE, T_END)) {
      // Don't save last state because we don't have a reconstruction for that and it messes with
      // the visualization
      if (t < T_END) {
        if (!data_writer.write(t)) { return 1; }
      }

      counter += 1;
    }
    get_vof_stats(vof.vf, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
    if (!check_vof(vof_min, vof_max, vof_integral, max_volume_error)) { return 1; }
    monitor.write();
  }
}
