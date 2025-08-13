#ifndef FLUID_SOLVER_VOF_HPP_
#define FLUID_SOLVER_VOF_HPP_

#include <fstream>

#include <Igor/StaticVector.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "IR.hpp"
#include "Operators.hpp"

template <typename Float, Index NX, Index NY>
struct VOF {
  InterfaceReconstruction<NX, NY> ir{};

  Matrix<Float, NX, NY> vf_old{};
  Matrix<Float, NX, NY> vf{};

  Matrix<Float, NX, NY> curv{};
};

static constexpr std::array CUBOID_OFFSETS{
    std::array<Index, 3>{1, 0, 0},  // 0
    std::array<Index, 3>{1, 1, 0},  // 1
    std::array<Index, 3>{1, 1, 1},  // 2
    std::array<Index, 3>{1, 0, 1},  // 3
    std::array<Index, 3>{0, 0, 0},  // 4
    std::array<Index, 3>{0, 1, 0},  // 5
    std::array<Index, 3>{0, 1, 1},  // 6
    std::array<Index, 3>{0, 0, 1},  // 7
};

static constexpr std::array FACE_VERTICES{
    std::array<IRL::UnsignedIndex_t, 4>{0, 1, 2, 3},  // 8
    std::array<IRL::UnsignedIndex_t, 4>{4, 5, 1, 0},  // 9
    std::array<IRL::UnsignedIndex_t, 4>{5, 6, 2, 1},  // 10
    std::array<IRL::UnsignedIndex_t, 4>{6, 7, 3, 2},  // 11
    std::array<IRL::UnsignedIndex_t, 4>{0, 3, 7, 4},  // 12
    std::array<IRL::UnsignedIndex_t, 4>{5, 4, 7, 6},  // 13
};

enum Dir : std::uint8_t { XM, XP, YM, YP, ZM, ZP };

static constexpr std::array FACE_DIRECTION{
    Dir::XP,  // 8
    Dir::ZM,  // 9
    Dir::YP,  // 10
    Dir::ZP,  // 11
    Dir::YM,  // 12
    Dir::XM,  // 13
};
static_assert(FACE_VERTICES.size() == FACE_DIRECTION.size());
static_assert(CUBOID_OFFSETS.size() + FACE_VERTICES.size() ==
                  14 /*IRL::Polyhedron24{}.getNumberOfVertices()*/,
              "Expected CUBOID_OFFSETS.size() + FACE_VERTICES.size() == "
              "IRL::Polyhedron24::getNumberOfVertices()");

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr auto advect_point(const IRL::Pt& pt,
                            const FS<Float, NX, NY>& fs,
                            const Matrix<Float, NX, NY>& Ui,
                            const Matrix<Float, NX, NY>& Vi,
                            Float dt) -> IRL::Pt {
  const auto [u1, v1] = eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, pt[0], pt[1]);
  const auto [u2, v2] =
      eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, pt[0] - 0.5 * dt * u1, pt[1] - 0.5 * dt * v1);
  const auto [u3, v3] =
      eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, pt[0] - 0.5 * dt * u2, pt[1] - 0.5 * dt * v2);
  const auto [u4, v4] = eval_flow_field_at(fs.xm, fs.ym, Ui, Vi, pt[0] - dt * u3, pt[1] - dt * v3);

  return {
      pt[0] - dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4),
      pt[1] - dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4),
      pt[2],
  };
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
constexpr auto advect_point2(const IRL::Pt& pt, const FS<Float, NX, NY>& fs, Float dt) -> IRL::Pt {
  const auto u1 = bilinear_interpolate(fs.x, fs.ym, fs.curr.U, pt[0], pt[1]);
  const auto v1 = bilinear_interpolate(fs.xm, fs.y, fs.curr.V, pt[0], pt[1]);

  const auto u2 =
      bilinear_interpolate(fs.x, fs.ym, fs.curr.U, pt[0] - 0.5 * dt * u1, pt[1] - 0.5 * dt * v1);
  const auto v2 =
      bilinear_interpolate(fs.xm, fs.y, fs.curr.V, pt[0] - 0.5 * dt * u1, pt[1] - 0.5 * dt * v1);

  const auto u3 =
      bilinear_interpolate(fs.x, fs.ym, fs.curr.U, pt[0] - 0.5 * dt * u2, pt[1] - 0.5 * dt * v2);
  const auto v3 =
      bilinear_interpolate(fs.xm, fs.y, fs.curr.V, pt[0] - 0.5 * dt * u2, pt[1] - 0.5 * dt * v2);

  const auto u4 = bilinear_interpolate(fs.x, fs.ym, fs.curr.U, pt[0] - dt * u3, pt[1] - dt * v3);
  const auto v4 = bilinear_interpolate(fs.xm, fs.y, fs.curr.V, pt[0] - dt * u3, pt[1] - dt * v3);

  return {
      pt[0] - dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4),
      pt[1] - dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4),
      pt[2],
  };
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
void reconstruct_interface(const FS<Float, NX, NY>& fs,
                           const Matrix<Float, NX, NY>& vf,
                           InterfaceReconstruction<NX, NY>& ir) {
  constexpr IRL::UnsignedIndex_t NEIGHBORHOOD_SIZE = 9;

  // Reset ir.interface
  std::fill_n(ir.interface.get_data(), ir.interface.size(), IRL::PlanarSeparator{});

  for (Index i = 0; i < vf.extent(0); ++i) {
    for (Index j = 0; j < vf.extent(1); ++j) {
      // Calculate the interface; skip if does not contain an interface
      if (!has_interface(vf, i, j)) { continue; }

      IRL::ELVIRANeighborhood neighborhood{};
      neighborhood.resize(NEIGHBORHOOD_SIZE);
      std::array<IRL::RectangularCuboid, NEIGHBORHOOD_SIZE> cells{};
      std::array<Float, NEIGHBORHOOD_SIZE> cells_vf{};

      size_t counter = 0;
      for (Index di = -1; di <= 1; ++di) {
        for (Index dj = -1; dj <= 1; ++dj) {
          const Float x_lo = fs.xm[i] + (di - 0.5) * fs.dx;
          const Float x_hi = fs.xm[i] + (di + 0.5) * fs.dx;
          const Float y_lo = fs.ym[j] + (dj - 0.5) * fs.dy;
          const Float y_hi = fs.ym[j] + (dj + 0.5) * fs.dy;
          const Float z_lo = -0.5;
          const Float z_hi = 0.5;

          cells[counter]   = IRL::RectangularCuboid::fromBoundingPts(IRL::Pt{x_lo, y_lo, z_lo},
                                                                   IRL::Pt{x_hi, y_hi, z_hi});
          // TODO: Assume that vf is always zero outside of the domain, i.e. Dirichlet Boundary
          //       conditions
          cells_vf[counter] = vf.is_valid_index(i + di, j + dj) ? vf[i + di, j + dj] : 0.0;
          neighborhood.setMember(&cells[counter], &cells_vf[counter], di, dj);
          counter += 1;
        }
      }
      ir.interface[i, j] = IRL::reconstructionWithELVIRA2D(neighborhood);
      IGOR_ASSERT((ir.interface[i, j].getNumberOfPlanes() == 1),
                  "({}, {}): Expected one plane but got {}",
                  i,
                  j,
                  (ir.interface[i, j].getNumberOfPlanes()));
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void advect_cells(const FS<Float, NX, NY>& fs,
                  [[maybe_unused]] const Matrix<Float, NX, NY>& Ui,
                  [[maybe_unused]] const Matrix<Float, NX, NY>& Vi,
                  Float dt,
                  VOF<Float, NX, NY>& vof,
                  Float* max_volume_error = nullptr) {
  constexpr Index NEIGHBORHOOD_OFFSET = 1;

  Float local_max_volume_error        = 0.0;
#pragma omp parallel for schedule(dynamic) collapse(2) reduction(max : local_max_volume_error)
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      // Early exit of loop iteration if we are entirely inside or outside of liquid phase
      {
        Float neighborhood_vf_sum = 0.0;
        for (Index ii = std::max(i - NEIGHBORHOOD_OFFSET, 0);
             ii < std::min(i + NEIGHBORHOOD_OFFSET + 1, NX);
             ++ii) {
          for (Index jj = std::max(j - NEIGHBORHOOD_OFFSET, 0);
               jj < std::min(j + NEIGHBORHOOD_OFFSET + 1, NY);
               ++jj) {
            neighborhood_vf_sum += vof.vf_old[ii, jj];
          }
        }
        if (neighborhood_vf_sum < VOF_LOW) { continue; }
        if (neighborhood_vf_sum >= Igor::sqr(2.0 * NEIGHBORHOOD_OFFSET + 1.0) * VOF_HIGH) {
          vof.vf[i, j] = 1.0;
          continue;
        }
      }

#ifndef VOF_NO_CORRECTION
      IRL::Polyhedron24 advected_cell{};
#else
      IRL::Dodecahedron advected_cell{};
#endif

      auto offset_to_pt = [i, j, &fs](Index di, Index dj, Index dk) {
        return IRL::Pt{fs.x[i + di], fs.y[j + dj], dk == 1 ? 0.5 : -0.5};
      };

      // = Set cuboid vertices =====================================================================
      for (IRL::UnsignedIndex_t cell_idx = 0; cell_idx < CUBOID_OFFSETS.size(); ++cell_idx) {
        const auto [di, dj, dk] = CUBOID_OFFSETS[cell_idx];
#ifndef FS_VOF_ADVECT_WITH_STAGGERED_VELOCITY
        advected_cell[cell_idx] = advect_point(offset_to_pt(di, dj, dk), fs, Ui, Vi, dt);
#else
        advected_cell[cell_idx] = advect_point2(offset_to_pt(di, dj, dk), fs, dt);
#endif  // FS_VOF_ADVECT_WITH_STAGGERED_VELOCITY
      }

#ifndef VOF_NO_CORRECTION
      // = Set other vertices to barycenter ========================================================
      for (IRL::UnsignedIndex_t cell_idx = 0; cell_idx < FACE_VERTICES.size(); ++cell_idx) {
        const auto& vertices = FACE_VERTICES[cell_idx];
        IRL::Pt barycenter   = IRL::Pt::fromScalarConstant(0.0);
        for (auto idx : vertices) {
          barycenter += advected_cell[idx];
        }
        barycenter                                      /= 4.0;
        advected_cell[cell_idx + CUBOID_OFFSETS.size()]  = barycenter;
      }
#endif  // VOF_NO_CORRECTION

      const auto original_cell_vol = (fs.x[i + 1] - fs.x[i]) * (fs.y[j + 1] - fs.y[j]);

#ifndef VOF_NO_CORRECTION
      // = Adjust other vertices to have correct cell volume =======================================
      for (IRL::UnsignedIndex_t cell_idx = 0; cell_idx < FACE_VERTICES.size(); ++cell_idx) {
        const auto& vertices = FACE_VERTICES[cell_idx];
        // We have a two dimensional problem, therefore we do not adjust the vertices in z-direction
        if (FACE_DIRECTION[cell_idx] == Dir::ZM || FACE_DIRECTION[cell_idx] == Dir::ZP) {
          continue;
        }

        const Float correct_flux_vol = [&] {
          switch (FACE_DIRECTION[cell_idx]) {
            case Dir::XM: return -fs.curr.U[i, j] * fs.dy * dt;
            case Dir::XP: return fs.curr.U[i + 1, j] * fs.dy * dt;
            case Dir::YM: return -fs.curr.V[i, j] * fs.dx * dt;
            case Dir::YP: return fs.curr.V[i, j + 1] * fs.dx * dt;
            case Dir::ZM:
            case Dir::ZP: Igor::Panic("Unreachable"); std::unreachable();
          }
        }();

        IRL::CappedDodecahedron advected_face{};
        IRL::UnsignedIndex_t counter = 0;
        for (auto idx : vertices) {
          // Vertices 0-3 are the ones of the original eulerian cell
          const auto [di, dj, dk] = CUBOID_OFFSETS[idx];
          advected_face[counter]  = offset_to_pt(di, dj, dk);

          // Vertices 4-7 are the advected vertices of the face
          const auto o                = advected_face.getNumberOfVertices() / 2;
          advected_face[counter + o]  = advected_cell[idx];

          counter                    += 1;
        }
        constexpr IRL::UnsignedIndex_t ADJUSTED_VERTEX = 8;
        // Barycenter of advected vertices is initial guess for vertex 8
        advected_face[ADJUSTED_VERTEX] = advected_cell[cell_idx + CUBOID_OFFSETS.size()];

        advected_face.adjustCapToMatchVolume(correct_flux_vol);
        advected_cell[cell_idx + CUBOID_OFFSETS.size()] = advected_face[ADJUSTED_VERTEX];
      }
#endif  // VOF_NO_CORRECTION

      local_max_volume_error =
          std::max(local_max_volume_error,
                   std::abs(original_cell_vol - advected_cell.calculateAbsoluteVolume()));

      Float overlap_vol = 0.0;
      for (Index ii = std::max(i - NEIGHBORHOOD_OFFSET, 0);
           ii < std::min(i + NEIGHBORHOOD_OFFSET + 1, NX);
           ++ii) {
        for (Index jj = std::max(j - NEIGHBORHOOD_OFFSET, 0);
             jj < std::min(j + NEIGHBORHOOD_OFFSET + 1, NY);
             ++jj) {
          if (vof.vf_old[ii, jj] > VOF_LOW) {
            overlap_vol += IRL::getVolumeMoments<IRL::Volume, IRL::RecursiveSimplexCutting>(
                advected_cell,
                IRL::LocalizedSeparator(&vof.ir.cell_localizer[ii, jj], &vof.ir.interface[ii, jj]));
          }
        }
      }

      vof.vf[i, j] = overlap_vol / advected_cell.calculateAbsoluteVolume();
    }
  }

  if (max_volume_error) { *max_volume_error = local_max_volume_error; }
}

// -------------------------------------------------------------------------------------------------
// template <typename Float, Index NX, Index NY>
// [[nodiscard]] constexpr auto get_interface_length(Index i,
//                                                   Index j,
//                                                   const Vector<Float, NX + 1>& x,
//                                                   const Vector<Float, NY + 1>& y,
//                                                   const IRL::Plane& plane) noexcept -> Float {
//   const auto [p1, p2] = get_intersections_with_cell<Float, NX, NY>(i, j, x, y, plane);
//   IGOR_ASSERT(
//       std::abs(p1[2]) < 1e-12, "Expected z-component of p1 to be zero but is {:.6e}", p1[2]);
//   IGOR_ASSERT(
//       std::abs(p2[2]) < 1e-12, "Expected z-component of p2 to be zero but is {:.6e}", p2[2]);
//   return std::sqrt(Igor::sqr(p1[0] - p2[0]) + Igor::sqr(p1[1] - p2[1]));
// }

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
[[nodiscard]] constexpr auto get_intersections_with_cell(Index i,
                                                         Index j,
                                                         const Vector<Float, NX + 1>& x,
                                                         const Vector<Float, NY + 1>& y,
                                                         const IRL::Plane& plane)
    -> std::array<IRL::Pt, 2> {
  Igor::StaticVector<IRL::Pt, 4> trial_points{};
  constexpr std::array offsets{
      std::pair<Index, Index>{0, 0},
      std::pair<Index, Index>{1, 0},
      std::pair<Index, Index>{1, 1},
      std::pair<Index, Index>{0, 1},
  };
  static_assert(offsets.size() == trial_points.max_size());

  constexpr Float EPS = 1e-6;
  for (size_t idx = 0; idx < offsets.size(); ++idx) {
    const auto [di0, dj0] = offsets[idx];
    IRL::Pt p0(x[i + di0], y[j + dj0], 0.0);

    const auto [di1, dj1] = offsets[(idx + 1) % offsets.size()];
    IRL::Pt p1(x[i + di1], y[j + dj1], 0.0);

    const auto tp = IRL::Pt::fromEdgeIntersection(
        p0, plane.signedDistanceToPoint(p0), p1, plane.signedDistanceToPoint(p1));

    if (x[i] - EPS <= tp[0] && tp[0] <= x[i + 1] + EPS &&  //
        y[j] - EPS <= tp[1] && tp[1] <= y[j + 1] + EPS) {
      trial_points.push_back(tp);
    }
  }
  IGOR_ASSERT(trial_points.size() >= 2,
              "Expected at least two trial points but got only {}.",
              trial_points.size());

  auto sqr_dist = [](const IRL::Pt& a, const IRL::Pt& b) {
    return Igor::sqr(b[0] - a[0]) + Igor::sqr(b[1] - a[1]);
  };
  IRL::Pt to_add0 = trial_points[0];
  IRL::Pt to_add1 = trial_points[1];
  Float distance  = sqr_dist(to_add0, to_add1);
  for (size_t add_idx0 = 0; add_idx0 < trial_points.size(); ++add_idx0) {
    for (size_t add_idx1 = add_idx0; add_idx1 < trial_points.size(); ++add_idx1) {
      const auto new_dist = sqr_dist(trial_points[add_idx0], trial_points[add_idx1]);
      if (new_dist > distance) {
        distance = new_dist;
        to_add0  = trial_points[add_idx0];
        to_add1  = trial_points[add_idx1];
      }
    }
  }

  return {to_add0, to_add1};
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
auto save_interface(const std::string& filename,
                    const Vector<Float, NX + 1>& x,
                    const Vector<Float, NY + 1>& y,
                    const Matrix<IRL::PlanarSeparator, NX, NY>& interface) -> bool {
  // - Find intersection points of planar separator and grid =======================================
  static std::vector<IRL::Pt> points{};
  points.resize(0);

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
      const IRL::Plane& plane  = interface[i, j][0];
      const auto intersections = get_intersections_with_cell<Float, NX, NY>(i, j, x, y, plane);
      static_assert(intersections.size() == 2);
      points.push_back(intersections[0]);
      points.push_back(intersections[1]);
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
  for (const auto& p : points) {
    const auto xi       = p[0];
    const auto yj       = p[1];
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
