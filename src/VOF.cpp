#include <optional>

// Disable warnings for IRL
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include <irl/geometry/general/pt.h>
#include <irl/interface_reconstruction_methods/elvira_neighborhood.h>
#include <irl/interface_reconstruction_methods/reconstruction_interface.h>
#pragma clang diagnostic pop

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>
#include <Igor/TypeName.hpp>

#include "Container.hpp"
#include "IO.hpp"
#include "Operators.hpp"

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
constexpr Float VOF_LOW     = 1e-8;
constexpr Float VOF_HIGH    = 1.0 - VOF_LOW;
Float INIT_VOF_INT          = 0.0;  // NOLINT

constexpr Float DT    = 5e-3;
constexpr Index NITER = 250;

constexpr size_t NEIGHBORHOOD_SIZE = 9;

constexpr auto OUTPUT_DIR = "output/VOF";
// = Config ========================================================================================

struct InterfaceReconstruction {
  Matrix<IRL::PlanarSeparator, NX, NY> interface;
  Matrix<IRL::PlanarLocalizer, NX, NY> cell_localizer;
};

constexpr auto has_interface(Float vof) noexcept -> bool { return VOF_LOW < vof && vof < VOF_HIGH; }

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
auto save_interface(const std::string& filename,
                    const Vector<Float, NX + 1>& x,
                    const Vector<Float, NY + 1>& y,
                    const Matrix<IRL::PlanarSeparator, NX, NY>& interface) -> bool {
  // Find intersection points of planar separator and grid
  static std::vector<std::pair<Float, Float>> points{};
  points.resize(0);

  auto get_x = [](IRL::Normal normal, Float dist, Float y) -> std::optional<Float> {
    IGOR_ASSERT(std::abs(normal[2]) < 1e-8,
                "Expect normal to not have a z-component, but got ({}, {}, {})",
                normal[0],
                normal[1],
                normal[2]);
    if (std::abs(normal[0]) < 1e-6) { return {}; }
    return std::optional{-normal[1] / normal[0] * y + dist / normal[0]};
  };

  auto get_y = [](IRL::Normal normal, Float dist, Float x) -> std::optional<Float> {
    IGOR_ASSERT(std::abs(normal[2]) < 1e-8,
                "Expect normal to not have a z-component, but got ({}, {}, {})",
                normal[0],
                normal[1],
                normal[2]);
    if (std::abs(normal[1]) < 1e-6) { return {}; }
    return std::optional{-normal[0] / normal[1] * x + dist / normal[1]};
  };

  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (interface[i, j].getNumberOfPlanes() != 1) {
        IGOR_ASSERT((interface[i, j].getNumberOfPlanes() == 0),
                    "{}, {}: Number of planes can only be 0 or 1 but is {}",
                    i,
                    j,
                    interface[i, j].getNumberOfPlanes());
        continue;
      }

      const auto begin_idx = points.size();
      for (Index idx = i; idx < i + 2; ++idx) {
        const Float x_trial = x[idx];
        const auto y_trial =
            get_y(interface[i, j][0].normal(), interface[i, j][0].distance(), x_trial);
        if (y_trial.has_value() && *y_trial > y[j] - 1e-4 && *y_trial < y[j + 1] + 1e-4 &&
            std::none_of(
                std::next(std::cbegin(points), static_cast<std::ptrdiff_t>(begin_idx)),
                std::cend(points),
                [x_trial, y_trial](const std::pair<Float, Float>& p) {
                  return std::sqrt(Igor::sqr(p.first - x_trial) + Igor::sqr(p.second - *y_trial)) <
                         1e-6;
                })) {
          points.emplace_back(x_trial, *y_trial);
        }
      }

      for (Index idx = j; idx < j + 2; ++idx) {
        const Float y_trial = y[idx];
        const auto x_trial =
            get_x(interface[i, j][0].normal(), interface[i, j][0].distance(), y_trial);
        if (x_trial.has_value() && *x_trial > x[i] - 1e-4 && *x_trial < x[i + 1] + 1e-4 &&
            std::none_of(
                std::next(std::cbegin(points), static_cast<std::ptrdiff_t>(begin_idx)),
                std::cend(points),
                [x_trial, y_trial](const std::pair<Float, Float>& p) {
                  return std::sqrt(Igor::sqr(p.first - *x_trial) + Igor::sqr(p.second - y_trial)) <
                         1e-6;
                })) {
          points.emplace_back(*x_trial, y_trial);
        }
      }
      // Just ignore edge cases
      if (points.size() - begin_idx != 2) { points.resize(begin_idx); }
      // IGOR_ASSERT(points.size() - begin_idx == 2,
      //             "{}, {}: Expected to add exactly two points to array but added {}: {}",
      //             i,
      //             j,
      //             points.size() - begin_idx,
      //             std::span(points.data() + begin_idx, points.size() - begin_idx));
    }
  }
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
  out << "DATASET POLYDATA\n";
  out << "POINTS " << points.size() << " double\n";
  for (const auto [xi, yj] : points) {
    constexpr double zk = 0.0;
    out.write(detail::interpret_as_big_endian_bytes(xi).data(), sizeof(xi));
    out.write(detail::interpret_as_big_endian_bytes(yj).data(), sizeof(yj));
    out.write(detail::interpret_as_big_endian_bytes(zk).data(), sizeof(zk));
  }
  out << "\n\n";

  // = Write cell data =============================================================================
  out << "LINES " << 3 << ' ' << points.size() / 2 * 3 << '\n';
  for (uint32_t i = 0; i < points.size(); i += 2) {
    out.write(detail::interpret_as_big_endian_bytes(uint32_t{2}).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i + 1).data(), sizeof(uint32_t));
  }

  return out.good();
}

// -------------------------------------------------------------------------------------------------
constexpr auto advect_point(const IRL::Pt& point, Float u, Float v) -> IRL::Pt {
  return IRL::Pt{point[0] - DT * u, point[1] - DT * v, point[2]};
}

// -------------------------------------------------------------------------------------------------
void print_vof_stats(const Matrix<Float, NX, NY>& vof) noexcept {
  const auto [min, max] = std::minmax_element(vof.get_data(), vof.get_data() + vof.size());
  const auto integral =
      std::reduce(vof.get_data(), vof.get_data() + vof.size(), 0.0, std::plus<>{}) * DX * DY;
  Igor::Info("min(vof)  = {:.6e}", *min);
  Igor::Info("max(vof)  = {:.6e}", *max);
  Igor::Info("int(vof)  = {:.6e}", integral);
  Igor::Info("loss(vof) = {:.6e} ({:.4f}%)",
             INIT_VOF_INT - integral,
             100.0 * (INIT_VOF_INT - integral) / INIT_VOF_INT);
  std::cout << "--------------------------------------------------------------------------------\n";
}

// -------------------------------------------------------------------------------------------------
void reconstruct_interface(const Vector<Float, NX + 1>& x,
                           const Vector<Float, NY + 1>& y,
                           const Matrix<Float, NX, NY>& vof,
                           InterfaceReconstruction& ir) {
  // Reset ir.interface
  std::fill_n(ir.interface.get_data(), ir.interface.size(), IRL::PlanarSeparator{});

  for (Index i = 1; i < vof.extent(0) - 1; ++i) {
    for (Index j = 1; j < vof.extent(1) - 1; ++j) {
      // Calculate the interface; skip if does not contain an interface
      if (!has_interface(vof[i, j])) { continue; }

      IRL::ELVIRANeighborhood neighborhood{};
      neighborhood.resize(NEIGHBORHOOD_SIZE);
      std::array<IRL::RectangularCuboid, NEIGHBORHOOD_SIZE> cells{};
      std::array<Float, NEIGHBORHOOD_SIZE> cells_vof{};

      size_t counter = 0;
      for (Index di = -1; di <= 1; ++di) {
        for (Index dj = -1; dj <= 1; ++dj) {
          cells[counter] = IRL::RectangularCuboid::fromBoundingPts(
              IRL::Pt{x[i + di], y[j + dj], -0.5}, IRL::Pt{x[i + di + 1], y[j + dj + 1], 0.5});
          cells_vof[counter] = vof[i + di, j + dj];
          neighborhood.setMember(&cells[counter], &cells_vof[counter], di, dj);
          counter += 1;
        }
      }
      ir.interface[i, j] = IRL::reconstructionWithELVIRA2D(neighborhood);
      IGOR_ASSERT((ir.interface[i, j].getNumberOfPlanes() == 1),
                  "({}, {}): Expected one planar but got {}",
                  i,
                  j,
                  (ir.interface[i, j].getNumberOfPlanes()));
    }
  }
}

// -------------------------------------------------------------------------------------------------
void advect_cells(const Vector<Float, NX + 1>& x,
                  const Vector<Float, NY + 1>& y,
                  const Matrix<Float, NX, NY>& vof,
                  const Matrix<Float, NX + 1, NY>& U,
                  const Matrix<Float, NX, NY + 1>& V,
                  const InterfaceReconstruction& ir,
                  Matrix<Float, NX, NY>& vof_next) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      IRL::Dodecahedron advected_cell{};
      // IRL::Hexahedron advected_cell{};

      constexpr std::array offsets{
          std::array<Index, 3>{1, 0, 0},
          std::array<Index, 3>{1, 1, 0},
          std::array<Index, 3>{1, 1, 1},
          std::array<Index, 3>{1, 0, 1},
          std::array<Index, 3>{0, 0, 0},
          std::array<Index, 3>{0, 1, 0},
          std::array<Index, 3>{0, 1, 1},
          std::array<Index, 3>{0, 0, 1},
      };
      for (IRL::UnsignedIndex_t cell_idx = 0; cell_idx < advected_cell.getNumberOfVertices();
           ++cell_idx) {
        const auto [di, dj, dk] = offsets[cell_idx];
        const IRL::Pt pt{x[i + di], y[j + dj], dk == 1 ? 0.5 : -0.5};
        const Float U_advect = [&] {
          if (j + dj == 0) {
            return U[i + di, j + dj];
          } else if (j + dj == NY) {
            return U[i + di, j + dj - 1];
          } else {
            return (U[i + di, j + dj] + U[i + di, j + dj - 1]) / 2.0;
          }
        }();
        const Float V_advect = [&] {
          if (i + di == 0) {
            return V[i + di, j + dj];
          } else if (i + di == NX) {
            return V[i + di - 1, j + dj];
          } else {
            return (V[i + di, j + dj] + V[i + di - 1, j + dj]) / 2.0;
          }
        }();

        advected_cell[cell_idx] = advect_point(pt, U_advect, V_advect);
      }

      Float overlap_vol                   = 0.0;
      constexpr Index neighborhood_offset = 1;
      for (Index ii = std::max(i - neighborhood_offset, 0);
           ii < std::min(i + neighborhood_offset, NX);
           ++ii) {
        for (Index jj = std::max(j - neighborhood_offset, 0);
             jj < std::min(j + neighborhood_offset, NY);
             ++jj) {
          if (vof[ii, jj] > VOF_LOW) {
            overlap_vol += IRL::getVolumeMoments<IRL::Volume>(
                advected_cell,
                IRL::LocalizedSeparator(&ir.cell_localizer[ii, jj], &ir.interface[ii, jj]));
          }
        }
      }

      vof_next[i, j] = overlap_vol / advected_cell.calculateAbsoluteVolume();
    }
  }
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

  InterfaceReconstruction ir{};
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
  for (Index i = 0; i < vof.extent(0); ++i) {
    for (Index j = 0; j < vof.extent(1); ++j) {
      // Localize the cell for volume calculation
      constexpr std::array<IRL::Normal, 6> plane_normals = {
          IRL::Normal(-1.0, 0.0, 0.0),
          IRL::Normal(1.0, 0.0, 0.0),
          IRL::Normal(0.0, -1.0, 0.0),
          IRL::Normal(0.0, 1.0, 0.0),
          IRL::Normal(0.0, 0.0, -1.0),
          IRL::Normal(0.0, 0.0, 1.0),
      };
      auto& l = ir.cell_localizer[i, j];
      l.setNumberOfPlanes(6);
      l[0] = IRL::Plane(plane_normals[0], -x[i]);
      l[1] = IRL::Plane(plane_normals[1], x[i + 1]);
      l[2] = IRL::Plane(plane_normals[2], -y[j]);
      l[3] = IRL::Plane(plane_normals[3], y[j + 1]);
      l[4] = IRL::Plane(plane_normals[4], 0.5);
      l[5] = IRL::Plane(plane_normals[5], 0.5);
    }
  }
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
      V[i, j] = 0.5 * xm[i];
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

  Igor::Info("iter = {}", 0);
  print_vof_stats(vof);

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
    advect_cells(x, y, vof, U, V, ir, vof_next);
    std::copy_n(vof_next.get_data(), vof_next.size(), vof.get_data());

    // Don't save last state because we don't have a reconstruction for that and it messes with the
    // visualization
    if (iter < NITER - 1) {
      if (!save_vof_state(
              Igor::detail::format("{}/vof_{:06d}.vtk", OUTPUT_DIR, iter + 1), x, y, vof, Ui, Vi)) {
        return 1;
      }
    }
    Igor::Info("iter = {}", iter + 1);
    print_vof_stats(vof);
  }
}
