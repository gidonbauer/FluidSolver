#include <numeric>
#include <optional>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "VOF.hpp"

// = Config ========================================================================================
using Float           = double;
constexpr Index NX    = 128;
constexpr Index NY    = 128;
constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;
constexpr auto DX     = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY     = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Index VOF_NSAMPLE = 10;
Float INIT_VOF_INT          = 0.0;  // NOLINT

constexpr Float DT    = 5e-3;
constexpr Index NITER = 120;

constexpr auto OUTPUT_DIR = "test/output/LinearVelocityVOF";
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
auto main() -> int {
  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  Matrix<Float, NX + 1, NY> U{};
  Matrix<Float, NX, NY + 1> V{};
  Matrix<Float, NX, NY> Ui{};
  Matrix<Float, NX, NY> Vi{};

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> vof_next{};

  Vector<Float, NX + 1> x{};
  Vector<Float, NY + 1> y{};
  Vector<Float, NX> xm{};
  Vector<Float, NY> ym{};

  InterfaceReconstruction<NX, NY> ir{};
  // = Allocate memory =============================================================================

  // = Setup grid and cell localizers ==============================================================
  for (Index i = 0; i < x.extent(0); ++i) {
    x[i] = X_MIN + static_cast<Float>(i) * DX;
  }
  for (Index i = 0; i < xm.extent(0); ++i) {
    xm[i] = (x[i] + x[i + 1]) / 2.0;
  }
  for (Index j = 0; j < y.extent(0); ++j) {
    y[j] = Y_MIN + static_cast<Float>(j) * DY;
  }
  for (Index j = 0; j < ym.extent(0); ++j) {
    ym[j] = (y[j] + y[j + 1]) / 2.0;
  }

  // Localize the cells
  localize_cells(x, y, ir);
  // = Setup grid and cell localizers ==============================================================

  // = Setup velocity and vof field ================================================================
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      auto is_in = [](Float x, Float y) -> Float {
        return Igor::sqr(x - 0.25) + Igor::sqr(y - 0.25) <= Igor::sqr(0.125);
      };

      vof[i, j] = 0.0;
      for (Index ii = 1; ii <= VOF_NSAMPLE; ++ii) {
        for (Index jj = 1; jj <= VOF_NSAMPLE; ++jj) {
          const auto xi = x[i] + static_cast<Float>(ii) / static_cast<Float>(VOF_NSAMPLE + 1) * DX;
          const auto yj = y[j] + static_cast<Float>(jj) / static_cast<Float>(VOF_NSAMPLE + 1) * DY;
          vof[i, j] += is_in(xi, yj);
        }
      }
      vof[i, j] /= Igor::sqr(VOF_NSAMPLE);
    }
  }

  for (Index i = 0; i < U.extent(0); ++i) {
    for (Index j = 0; j < U.extent(1); ++j) {
      U[i, j] = ym[j];
    }
  }

  for (Index i = 0; i < V.extent(0); ++i) {
    for (Index j = 0; j < V.extent(1); ++j) {
      V[i, j] = xm[i];
    }
  }

  interpolate_U(U, Ui);
  interpolate_V(V, Vi);
  if (!save_vof_state(
          Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, 0), x, y, vof, Ui, Vi)) {
    return 1;
  }
  INIT_VOF_INT =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;
  // = Setup velocity and vof field ================================================================

  Igor::ScopeTimer timer("LinearVelocityVOF");
  Float max_volume_error = 0.0;
  for (Index iter = 0; iter < NITER; ++iter) {
    // = Reconstruct the interface =================================================================
    reconstruct_interface(x, y, vof, ir);
    if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                        x,
                        y,
                        ir.interface)) {
      return 1;
    }

    // = Advect cells ==============================================================================
    advect_cells(x, y, xm, ym, vof, Ui, Vi, DT, ir, vof_next, &max_volume_error);
    std::copy_n(vof_next.get_data(), vof_next.size(), vof.get_data());

    // Don't save last state because we don't have a reconstruction for that and it messes with the
    // visualization
    if (iter < NITER - 1) {
      if (!save_vof_state(
              Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, iter + 1), x, y, vof, Ui, Vi)) {
        return 1;
      }
    }
    if (max_volume_error > 5e-10) {
      Igor::Warn("Advected cells expanded: max. volume error = {:.6e}", max_volume_error);
      return 1;
    }
    if (!check_vof(vof)) { return 1; }
  }
}
