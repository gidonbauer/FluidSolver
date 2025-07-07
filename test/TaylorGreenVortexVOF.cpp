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

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 128;
constexpr Index NY    = 128;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 2.0 * std::numbers::pi_v<Float>;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 2.0 * std::numbers::pi_v<Float>;
constexpr auto DX     = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY     = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float VISC = 1e-1;
constexpr Float RHO  = 0.9;

Float INIT_VOF_INT = 0.0;  // NOLINT

constexpr Float DT_MAX   = 1e-2;
constexpr Float CFL_MAX  = 0.5;
constexpr Float T_END    = 5.0;
constexpr Float DT_WRITE = 5e-2;

constexpr auto OUTPUT_DIR = "test/output/TaylorGreenVortexVOF";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
auto save_vof_state(const std::string& filename,
                    const Vector<Float, NX + 1>& x,
                    const Vector<Float, NY + 1>& y,
                    const Matrix<Float, NX, NY>& vof,
                    const Matrix<Float, NX, NY>& Ui,
                    const Matrix<Float, NX, NY>& Vi) -> bool {
  std::ofstream out(filename);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  // = Write VTK header ============================================================================
  out << "# vtk DataFile Version 2.0\n";
  out << "VOF field\n";
  out << "BINARY\n";

  // = Write grid ==================================================================================
  out << "DATASET STRUCTURED_GRID\n";
  out << "DIMENSIONS " << x.size() << ' ' << y.size() << " 1\n";
  out << "POINTS " << x.size() * y.size() << " double\n";
  for (Index j = 0; j < y.size(); ++j) {
    for (Index i = 0; i < x.size(); ++i) {
      constexpr double zk = 0.0;
      out.write(detail::interpret_as_big_endian_bytes(x[i]).data(), sizeof(x[i]));
      out.write(detail::interpret_as_big_endian_bytes(y[j]).data(), sizeof(y[j]));
      out.write(detail::interpret_as_big_endian_bytes(zk).data(), sizeof(zk));
    }
  }
  out << "\n\n";

  // = Write cell data =============================================================================
  out << "CELL_DATA " << vof.size() << '\n';
  detail::write_scalar_vtk(out, vof, "VOF");
  detail::write_vector_vtk(out, Ui, Vi, "velocity");

  return out.good();
}

// -------------------------------------------------------------------------------------------------
void get_vof_stats(const Matrix<Float, NX, NY>& vof,
                   Float& min,
                   Float& max,
                   Float& integral,
                   Float& loss,
                   Float& loss_prct) noexcept {
  const auto [min_it, max_it] = std::minmax_element(vof.get_data(), vof.get_data() + vof.size());

  min      = *min_it;
  max      = *max_it;
  integral = std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;
  loss     = INIT_VOF_INT - integral;
  loss_prct = 100.0 * loss / INIT_VOF_INT;
}

// -------------------------------------------------------------------------------------------------
auto check_vof(Float vof_min, Float vof_max, Float vof_integral, Float max_volume_error) -> bool {
  if (std::abs(vof_min) > 1e-14) {
    Igor::Warn("Expected minimum VOF value to be 0 but is {:.6e}", vof_min);
    return false;
  }

  if (std::abs(vof_max - 1.0) > 1e-14) {
    Igor::Warn("Expected maximum VOF value to be 1 but is {:.6e}", vof_max);
    return false;
  }

  if (std::abs(vof_integral - INIT_VOF_INT) > 1e-10) {
    Igor::Warn("Expected integral of vof to be {:.6e} but is {:.6e}: error = {:.6e}",
               INIT_VOF_INT,
               vof_integral,
               std::abs(vof_integral - INIT_VOF_INT));
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

void constexpr set_velocity(const Vector<Float, NX + 1>& x,
                            const Vector<Float, NY + 1>& y,
                            const Vector<Float, NX>& xm,
                            const Vector<Float, NY>& ym,
                            Float t,
                            Matrix<Float, NX + 1, NY>& U,
                            Matrix<Float, NX, NY + 1>& V) {
  for (Index i = 0; i < U.extent(0); ++i) {
    for (Index j = 0; j < U.extent(1); ++j) {
      U[i, j] = u_analytical(x[i], ym[j], t);
    }
  }

  for (Index i = 0; i < V.extent(0); ++i) {
    for (Index j = 0; j < V.extent(1); ++j) {
      V[i, j] = v_analytical(xm[i], y[j], t);
    }
  }
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{.visc = VISC, .rho = RHO};
  InterfaceReconstruction<NX, NY> ir{};

  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};
  Matrix<Float, NX, NY> div{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_next{};

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  for (Index i = 0; i < fs.x.extent(0); ++i) {
    fs.x[i] = X_MIN + static_cast<Float>(i) * DX;
  }
  for (Index j = 0; j < fs.y.extent(0); ++j) {
    fs.y[j] = Y_MIN + static_cast<Float>(j) * DY;
  }
  init_mid_and_delta(fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, ir);
  // = Setup grid and cell localizers ==============================================================

  // = Setup velocity and vof field ================================================================
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      auto is_in = [](Float x, Float y) -> Float {
        return Igor::sqr(x - std::numbers::pi) + Igor::sqr(y - 1.5 * std::numbers::pi) <=
               Igor::sqr(0.5);
      };

      vof[i, j] =
          quadrature(is_in, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx[i] * fs.dy[j]);
    }
  }

  set_velocity(fs.x, fs.y, fs.xm, fs.ym, 0.0, fs.U, fs.V);
  interpolate_U(fs.U, Ui);
  interpolate_V(fs.V, Vi);
  calc_divergence(fs, div);
  Float max_div = std::transform_reduce(
      div.get_data(),
      div.get_data() + div.size(),
      0.0,
      [](Float a, Float b) { return std::max(a, b); },
      [](Float a) { return std::abs(a); });

  if (!save_vof_state(
          Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, 0), fs.x, fs.y, vof, Ui, Vi)) {
    return 1;
  }
  INIT_VOF_INT =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;
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

  get_vof_stats(vof, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
  if (!check_vof(vof_min, vof_max, vof_integral, max_volume_error)) { return 1; }

  Index counter = 0;
  Igor::ScopeTimer timer("TaylorGreenVortexVOF");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // = Reconstruct the interface =================================================================
    reconstruct_interface(fs.x, fs.y, vof, ir);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, counter),
                          fs.x,
                          fs.y,
                          ir.interface)) {
        return 1;
      }
    }

    // = Advect cells ==============================================================================
    advect_cells(fs, vof, Ui, Vi, dt, ir, vof_next, &max_volume_error);
    std::copy_n(vof_next.get_data(), vof_next.size(), vof.get_data());

    // = Update velocity field according to analytical solution ====================================
    t += dt;
    set_velocity(fs.x, fs.y, fs.xm, fs.ym, t, fs.U, fs.V);
    interpolate_U(fs.U, Ui);
    interpolate_V(fs.V, Vi);
    calc_divergence(fs, div);
    max_div = std::transform_reduce(
        div.get_data(),
        div.get_data() + div.size(),
        0.0,
        [](Float a, Float b) { return std::max(a, b); },
        [](Float a) { return std::abs(a); });

    if (should_save(t, dt, DT_WRITE, T_END)) {
      // Don't save last state because we don't have a reconstruction for that and it messes with
      // the visualization
      if (t < T_END) {
        if (!save_vof_state(Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, counter + 1),
                            fs.x,
                            fs.y,
                            vof,
                            Ui,
                            Vi)) {
          return 1;
        }
      }

      counter += 1;
    }
    get_vof_stats(vof, vof_min, vof_max, vof_integral, vof_loss, vof_loss_prct);
    if (!check_vof(vof_min, vof_max, vof_integral, max_volume_error)) { return 1; }
    monitor.write();
  }
}
