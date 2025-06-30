#ifndef FLUID_SOLVER_VOF_HPP_
#define FLUID_SOLVER_VOF_HPP_

#include <fstream>

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

#include "Container.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"

constexpr double VOF_LOW  = 1e-8;
constexpr double VOF_HIGH = 1.0 - VOF_LOW;

template <typename Float>
constexpr auto has_interface(Float vof) noexcept -> bool {
  return VOF_LOW < vof && vof < VOF_HIGH;
}

template <Index NX, Index NY>
struct InterfaceReconstruction {
  Matrix<IRL::PlanarSeparator, NX, NY> interface;
  Matrix<IRL::PlanarLocalizer, NX, NY> cell_localizer;
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr auto advect_point(const IRL::Pt& pt,
                            const Vector<Float, NX>& xm,
                            const Vector<Float, NY>& ym,
                            const Matrix<Float, NX, NY>& Ui,
                            const Matrix<Float, NX, NY>& Vi,
                            Float dt) -> IRL::Pt {
#ifdef ADVECT_EULER
  const auto [u, v] = eval_flow_field_at(xm, ym, Ui, Vi, pt[0], pt[1]);
  return {pt[0] - dt * u, pt[1] - dt * v, pt[2]};
#elif defined(ADVECT_RK2)
  const auto [u1, v1] = eval_flow_field_at(xm, ym, Ui, Vi, pt[0], pt[1]);
  const auto [u2, v2] =
      eval_flow_field_at(xm, ym, Ui, Vi, pt[0] - dt / 2.0 * u1, pt[1] - dt / 2.0 * v1);
  return {pt[0] - dt * u2, pt[1] - dt * v2, pt[2]};
#else
  const auto [u1, v1] = eval_flow_field_at(xm, ym, Ui, Vi, pt[0], pt[1]);
  const auto [u2, v2] =
      eval_flow_field_at(xm, ym, Ui, Vi, pt[0] - 0.5 * dt * u1, pt[1] - 0.5 * dt * v1);
  const auto [u3, v3] =
      eval_flow_field_at(xm, ym, Ui, Vi, pt[0] - 0.5 * dt * u2, pt[1] - 0.5 * dt * v2);
  const auto [u4, v4] = eval_flow_field_at(xm, ym, Ui, Vi, pt[0] - dt * u3, pt[1] - dt * v3);

  return {
      pt[0] - dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4),
      pt[1] - dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4),
      pt[2],
  };
#endif
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void localize_cells(const Vector<Float, NX + 1>& x,
                    const Vector<Float, NY + 1>& y,
                    InterfaceReconstruction<NX, NY>& ir) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
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
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void reconstruct_interface(const Vector<Float, NX + 1>& x,
                           const Vector<Float, NY + 1>& y,
                           const Matrix<Float, NX, NY>& vof,
                           InterfaceReconstruction<NX, NY>& ir) {
  constexpr IRL::UnsignedIndex_t NEIGHBORHOOD_SIZE = 9;

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
template <typename Float, Index NX, Index NY>
void advect_cells(const Vector<Float, NX + 1>& x,
                  const Vector<Float, NY + 1>& y,
                  const Vector<Float, NX>& xm,
                  const Vector<Float, NY>& ym,
                  const Matrix<Float, NX, NY>& vof,
                  const Matrix<Float, NX, NY>& Ui,
                  const Matrix<Float, NX, NY>& Vi,
                  Float dt,
                  const InterfaceReconstruction<NX, NY>& ir,
                  Matrix<Float, NX, NY>& vof_next,
                  Monitor<Float>* monitor = nullptr) {
  Float max_volume_error = 0.0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      // TODO: Use IRL::Polyhedron24 with correction to conserve vof in linear velocity fields
      IRL::Dodecahedron advected_cell{};

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
        advected_cell[cell_idx] = advect_point(pt, xm, ym, Ui, Vi, dt);
      }

      const auto original_cell_vol = (x[i + 1] - x[i]) * (y[j + 1] - y[j]);
      max_volume_error             = std::max(
          max_volume_error, std::abs(original_cell_vol - advected_cell.calculateAbsoluteVolume()));

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

  if (monitor) { monitor->max_volume_error = max_volume_error; }
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto save_cells(const std::string& filename,
                              const std::vector<std::array<IRL::Pt, 4>>& cells) -> bool {
  std::ofstream out(filename);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  // = Write VTK header ============================================================================
  out << "# vtk DataFile Version 2.0\n";
  out << "Advected cells\n";
  out << "BINARY\n";

  // = Write grid ==================================================================================
  out << "DATASET POLYDATA\n";
  out << "POINTS " << cells.size() * 4 << " double\n";
  for (const auto& cell : cells) {
    for (const auto& pt : cell) {
      static_assert(std::is_same_v<std::remove_cvref_t<decltype(pt[0])>, double>);
      constexpr double zk = 0.0;
      out.write(detail::interpret_as_big_endian_bytes(pt[0]).data(), sizeof(pt[0]));
      out.write(detail::interpret_as_big_endian_bytes(pt[1]).data(), sizeof(pt[1]));
      out.write(detail::interpret_as_big_endian_bytes(zk).data(), sizeof(zk));
    }
  }
  out << "\n\n";

  // = Write cell data =============================================================================
  out << "LINES " << 3 << ' ' << cells.size() * 5 << '\n';
  for (uint32_t i = 0; i < cells.size() * 4; i += 4) {
    out.write(detail::interpret_as_big_endian_bytes(uint32_t{4}).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i + 0).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i + 1).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i + 2).data(), sizeof(uint32_t));
    out.write(detail::interpret_as_big_endian_bytes(i + 3).data(), sizeof(uint32_t));
  }

  return out.good();
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
auto save_interface(const std::string& filename,
                    const Vector<Float, NX + 1>& x,
                    const Vector<Float, NY + 1>& y,
                    const Matrix<IRL::PlanarSeparator, NX, NY>& interface) -> bool {
  // - Find intersection points of planar separator and grid =======================================
  // TODO: Implement a proper algorithm for this and handle the edge cases
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
        if (y_trial.has_value() && *y_trial > y[j] - 1e-4 && *y_trial < y[j + 1] + 1e-4) {
          points.emplace_back(x_trial, *y_trial);
        }
      }

      for (Index idx = j; idx < j + 2; ++idx) {
        const Float y_trial = y[idx];
        const auto x_trial =
            get_x(interface[i, j][0].normal(), interface[i, j][0].distance(), y_trial);
        if (x_trial.has_value() && *x_trial > x[i] - 1e-4 && *x_trial < x[i + 1] + 1e-4) {
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

#endif  // FLUID_SOLVER_VOF_HPP_
