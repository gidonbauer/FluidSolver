#include <numeric>
#include <optional>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"

// = Config ========================================================================================
using Float               = double;
constexpr Index NX        = 128;
constexpr Index NY        = 128;
constexpr Float X_MIN     = 0.0;
constexpr Float X_MAX     = 1.0;
constexpr Float Y_MIN     = 0.0;
constexpr Float Y_MAX     = 1.0;
constexpr auto DX         = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY         = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float U0        = 1.0;
constexpr Float V0        = 0.5;

Float INIT_VOF_INT        = 0.0;  // NOLINT

constexpr Float DT        = 5e-3;
constexpr Index NITER     = 120;

constexpr auto OUTPUT_DIR = "test/output/ConstantVelocityVOF";
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
auto check_vof(const Matrix<Float, NX, NY>& vof) noexcept -> bool {
  const auto [min, max] = std::minmax_element(vof.get_data(), vof.get_data() + vof.size());
  const auto integral =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;

  constexpr Float EPS = 1e-12;
  if (std::abs(*min) > EPS) {
    Igor::Warn("Expected minimum VOF value to be 0 but is {:.6e}", *min);
    return false;
  }

  if (std::abs(*max - 1.0) > EPS) {
    Igor::Warn("Expected maximum VOF value to be 1 but is {:.6e}", *max);
    return false;
  }

  if (std::abs(integral - INIT_VOF_INT) > EPS) {
    Igor::Warn("Expected integral of vof to be {:.6e} but is {:.6e}: error = {:.6e}",
               INIT_VOF_INT,
               integral,
               std::abs(integral - INIT_VOF_INT));
    return false;
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
auto calc_center_of_mass(const Vector<Float, NX>& xm,
                         const Vector<Float, NY>& ym,
                         const Matrix<Float, NX, NY>& vof) -> std::array<Float, 2> {
  const auto mass =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;

  Float weighted_x = 0.0;
  Float weighted_y = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      weighted_x += xm[i] * vof[i, j];
      weighted_y += ym[j] * vof[i, j];
    }
  }
  weighted_x *= DX * DY;
  weighted_y *= DX * DY;

  return {weighted_x / mass, weighted_y / mass};
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_next{};

  InterfaceReconstruction<NX, NY> ir{};
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, ir);
  // = Setup grid and cell localizers ==============================================================

  // = Setup velocity and vof field ================================================================
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      auto is_in = [](Float x, Float y) -> Float {
        return Igor::sqr(x - 0.25) + Igor::sqr(y - 0.25) <= Igor::sqr(0.125);
      };

      vof[i, j] = quadrature(is_in, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx * fs.dy);
    }
  }

  for (Index i = 0; i < fs.curr.U.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.U.extent(1); ++j) {
      fs.curr.U[i, j] = U0;
    }
  }

  for (Index i = 0; i < fs.curr.V.extent(0); ++i) {
    for (Index j = 0; j < fs.curr.V.extent(1); ++j) {
      fs.curr.V[i, j] = V0;
    }
  }

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  if (!save_vof_state(
          Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, 0), fs.x, fs.y, vof, Ui, Vi)) {
    return 1;
  }
  INIT_VOF_INT =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;

  const std::array<Float, 2> init_center_of_mass = calc_center_of_mass(fs.xm, fs.ym, vof);
  // = Setup velocity and vof field ================================================================

  Igor::ScopeTimer timer("ConstantVelocityVOF");
  Float max_volume_error = 0.0;
  for (Index iter = 0; iter < NITER; ++iter) {
    // = Reconstruct the interface =================================================================
    reconstruct_interface(fs.x, fs.y, vof, ir);
    if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                        fs.x,
                        fs.y,
                        ir.interface)) {
      return 1;
    }

    // = Advect cells ==============================================================================
    advect_cells(fs, vof, Ui, Vi, DT, ir, vof_next, &max_volume_error);
    std::copy_n(vof_next.get_data(), vof_next.size(), vof.get_data());

    const auto center_of_mass = calc_center_of_mass(fs.xm, fs.ym, vof);
    const std::array<Float, 2> expected_center_of_mass{
        init_center_of_mass[0] + static_cast<Float>(iter + 1) * DT * U0,
        init_center_of_mass[1] + static_cast<Float>(iter + 1) * DT * V0,
    };
    const auto distance = std::sqrt(Igor::sqr(center_of_mass[0] - expected_center_of_mass[0]) +
                                    Igor::sqr(center_of_mass[1] - expected_center_of_mass[1]));
    if (distance > 5e-5) {
      Igor::Warn(
          "Center of mass did not move with the velocity field, is at position ({:.6e}, {:.6e}) "
          "but expected it to be at ({:.6e}, {:.6e}): distance = {:.6e}",
          center_of_mass[0],
          center_of_mass[1],
          expected_center_of_mass[0],
          expected_center_of_mass[1],
          distance);
      return 1;
    }

    // Don't save last state because we don't have a reconstruction for that and it messes with the
    // visualization
    if (iter < NITER - 1) {
      if (!save_vof_state(Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, iter + 1),
                          fs.x,
                          fs.y,
                          vof,
                          Ui,
                          Vi)) {
        return 1;
      }
    }
    if (max_volume_error > 1e-12) {
      Igor::Warn("Advected cells expanded: max. volume error = {:.6e}", max_volume_error);
      return 1;
    }
    if (!check_vof(vof)) { return 1; }
  }
}
