#ifndef FLUID_SOLVER_VOF_HPP_
#define FLUID_SOLVER_VOF_HPP_

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
template <typename Float>
constexpr auto advect_point(const IRL::Pt& point, Float u, Float v, Float dt) -> IRL::Pt {
  return {point[0] - dt * u, point[1] - dt * v, point[2]};
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
                  const Matrix<Float, NX, NY>& vof,
                  const Matrix<Float, NX + 1, NY>& U,
                  const Matrix<Float, NX, NY + 1>& V,
                  Float dt,
                  const InterfaceReconstruction<NX, NY>& ir,
                  Matrix<Float, NX, NY>& vof_next) {
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

        advected_cell[cell_idx] = advect_point(pt, U_advect, V_advect, dt);
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

#endif  // FLUID_SOLVER_VOF_HPP_
