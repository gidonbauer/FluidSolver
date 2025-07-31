#include <numbers>
#include <random>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/ProgressBar.hpp>
#include <Igor/StaticVector.hpp>
#include <Igor/Timer.hpp>

// #define FS_CURV_NO_INTERPOLATION
#include "Curvature.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                   = double;
constexpr Index NX            = 128;
constexpr Index NY            = 128;
constexpr Float X_MIN         = 0.0;
constexpr Float X_MAX         = 1.0;
constexpr Float Y_MIN         = 0.0;
constexpr Float Y_MAX         = 1.0;

constexpr Index NUM_TEST_ITER = 1;  // 1000;

constexpr auto OUTPUT_DIR     = "output/Curvature";
// = Config ========================================================================================

std::ofstream* quad_out  = nullptr;
std::ofstream* quad2_out = nullptr;

// -------------------------------------------------------------------------------------------------
enum : size_t { X, Y, NDIMS };
struct Interface {
  std::array<Float, NDIMS> begin{};
  std::array<Float, NDIMS> end{};
  std::array<Float, NDIMS> normal{};
};

// -------------------------------------------------------------------------------------------------
auto extract_interface(Index i,
                       Index j,
                       const FS<Float, NX, NY>& fs,
                       const InterfaceReconstruction<NX, NY>& ir) -> Interface {
  IGOR_ASSERT((ir.interface[i, j].getNumberOfPlanes() == 1),
              "Expected exactly one plane but got {}",
              (ir.interface[i, j].getNumberOfPlanes()));
  const auto& plane    = ir.interface[i, j][0];
  const auto intersect = get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, plane);

  IGOR_ASSERT(std::abs(plane.normal()[NDIMS]) < 1e-12,
              "Expected z-component of normal to be zero but is {:.6e}",
              plane.normal()[NDIMS]);
  return {
      .begin  = {intersect[0][X], intersect[0][Y]},
      .end    = {intersect[1][X], intersect[1][Y]},
      .normal = {plane.normal()[X], plane.normal()[Y]},
  };
}

// -------------------------------------------------------------------------------------------------
template <size_t CAPACITY>
void rotate_interfaces(Igor::StaticVector<Interface, CAPACITY>& interfaces) {
  IGOR_ASSERT(
      interfaces.size() >= 1, "Expected at least one interface but got {}", interfaces.size());

  auto angle = std::acos(-interfaces[0].normal[Y]);
  if (interfaces[0].normal[X] > 0.0) { angle = 2.0 * std::numbers::pi - angle; }
  auto rotate = [angle](const std::array<Float, NDIMS>& vec) -> std::array<Float, NDIMS> {
    return {
        std::cos(angle) * vec[X] - std::sin(angle) * vec[Y],
        std::sin(angle) * vec[X] + std::cos(angle) * vec[Y],
    };
  };

  for (auto& i : interfaces) {
    i.begin  = rotate(i.begin);
    i.end    = rotate(i.end);
    i.normal = rotate(i.normal);
  }

#if 1
  const std::array<Float, NDIMS> center = {
      (interfaces[0].begin[X] + interfaces[0].end[X]) / 2.0,
      (interfaces[0].begin[Y] + interfaces[0].end[Y]) / 2.0,
  };
  for (auto& i : interfaces) {
    i.begin[X] -= center[X];
    i.begin[Y] -= center[Y];
    i.end[X]   -= center[X];
    i.end[Y]   -= center[Y];
  }
#endif
}

// -------------------------------------------------------------------------------------------------
template <size_t CAPACITY>
void sort_begin_end(Igor::StaticVector<Interface, CAPACITY>& interfaces) {
  for (auto& i : interfaces) {
    if (i.begin[X] > i.end[X]) { std::swap(i.begin, i.end); }
  }
}

// -------------------------------------------------------------------------------------------------
void solve_linear_system(const Matrix<Float, 3, 3>& lhs,
                         const Vector<Float, 3>& rhs,
                         Vector<Float, 3>& sol) noexcept {
  Matrix<Float, 3, 3> inv_lhs{};
  inv_lhs[0, 0] = (lhs[1, 1] * lhs[2, 2] - lhs[1, 2] * lhs[2, 1]);
  inv_lhs[1, 0] = -(lhs[1, 0] * lhs[2, 2] - lhs[1, 2] * lhs[2, 0]);
  inv_lhs[2, 0] = (lhs[1, 0] * lhs[2, 1] - lhs[1, 1] * lhs[2, 0]);

  inv_lhs[0, 1] = -(lhs[0, 1] * lhs[2, 2] - lhs[0, 2] * lhs[2, 1]);
  inv_lhs[1, 1] = (lhs[0, 0] * lhs[2, 2] - lhs[0, 2] * lhs[2, 0]);
  inv_lhs[2, 1] = -(lhs[0, 0] * lhs[2, 1] - lhs[0, 1] * lhs[2, 0]);

  inv_lhs[0, 2] = (lhs[0, 1] * lhs[1, 2] - lhs[0, 2] * lhs[1, 1]);
  inv_lhs[1, 2] = -(lhs[0, 0] * lhs[1, 2] - lhs[0, 2] * lhs[1, 0]);
  inv_lhs[2, 2] = (lhs[0, 0] * lhs[1, 1] - lhs[0, 1] * lhs[1, 0]);

  const auto det_lhs =
      lhs[0, 0] * inv_lhs[0, 0] + lhs[0, 1] * inv_lhs[1, 0] + lhs[0, 2] * inv_lhs[2, 0];
  // IGOR_ASSERT(
  //     std::abs(det_lhs) > 1e-6, "Expected invertible matrix but determinant is {:.6e}", det_lhs);
  std::for_each_n(
      inv_lhs.get_data(), inv_lhs.size(), [det_lhs](Float& value) { value /= det_lhs; });

  for (Index i = 0; i < 3; ++i) {
    sol[i] = 0.0;
    for (Index j = 0; j < 3; ++j) {
      sol[i] += inv_lhs[i, j] * rhs[j];
    }
  }
}

// -------------------------------------------------------------------------------------------------
auto calc_cuvature_of_quad_poly(Float x, const Vector<Float, 3>& c) {
  const auto first_der  = c[1] + 2.0 * c[2] * x;
  const auto second_der = 2.0 * c[2];
  return second_der / std::pow(1.0 + Igor::sqr(first_der), 1.5);
}

// -------------------------------------------------------------------------------------------------
template <size_t CAPACITY>
auto calc_curv_quad_reconstruct(const Igor::StaticVector<Interface, CAPACITY>& interfaces)
    -> Float {
  Igor::StaticVector<std::array<Float, NDIMS>, 9UZ> plic_param{};
  for (const auto& i : interfaces) {
    const auto b1 = (i.end[Y] - i.begin[Y]) / (i.end[X] - i.begin[X]);
    const auto b0 = i.begin[Y] - b1 * i.begin[X];
    plic_param.push_back({b0, b1});
  }

  Igor::StaticVector<std::array<Float, 3UZ>, 9UZ> S{};
  for (const auto& i : interfaces) {
    S.push_back({
        i.end[X] - i.begin[X],
        1.0 / 2.0 * (std::pow(i.end[X], 2.0) - std::pow(i.begin[X], 2.0)),
        1.0 / 3.0 * (std::pow(i.end[X], 3.0) - std::pow(i.begin[X], 3.0)),
    });
  }

  Matrix<Float, 3, 3> A{};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      for (size_t r = 0; r < interfaces.size(); ++r) {
        A[i, j] += S[r][static_cast<size_t>(i)] * S[r][static_cast<size_t>(j)];
      }
    }
  }

  Vector<Float, 3> d{};
  for (Index i = 0; i < 3; ++i) {
    for (size_t r = 0; r < interfaces.size(); ++r) {
      d[i] +=
          S[r][static_cast<size_t>(i)] * (plic_param[r][0] * S[r][0] + plic_param[r][1] * S[r][1]);
    }
  }

  Vector<Float, 3> c{};
  solve_linear_system(A, d, c);
  IGOR_ASSERT(
      std::abs(interfaces[0].begin[Y] - interfaces[0].end[Y]) < 1e-8,
      "Expected target interface to be a horizontal line but begin[Y] = {:.6e} and end[Y] = {:.6e}",
      interfaces[0].begin[Y],
      interfaces[0].end[Y]);

  const auto eval_point = (interfaces[0].begin[X] + interfaces[0].end[X]) / 2.0;
  if (quad_out != nullptr) {
    for (size_t r = 0; r < interfaces.size(); ++r) {
      const auto& i = interfaces[r];
      (*quad_out) << Igor::detail::format("interface {} {:.6e} {:.6e} {:.6e} {:.6e}\n",
                                          r,
                                          i.begin[X],
                                          i.begin[Y],
                                          i.end[X],
                                          i.end[Y]);
    }
    (*quad_out) << Igor::detail::format("quad {:.6e} {:.6e} {:.6e}\n", c[0], c[1], c[2]);
    (*quad_out) << Igor::detail::format("eval_point {:.6e}\n", eval_point);
    (*quad_out) << "------------------------------------------------------------\n";
  }

  return calc_cuvature_of_quad_poly(eval_point, c);
}

// -------------------------------------------------------------------------------------------------
// Reference: A Paraboloid Fitting Technique for  Calculating Curvature from Piecewise-Linear
//            Interface Reconstructions on 3D  Unstructured Meshes by Z. Jibben, N. N. Carlson, and
//            M. M. Francois
void curvature_from_quad_reconstruction(const FS<Float, NX, NY>& fs,
                                        const InterfaceReconstruction<NX, NY>& ir,
                                        const Matrix<Float, NX, NY>& vof,
                                        Matrix<Float, NX, NY>& curv) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        Igor::StaticVector<Interface, 27UZ> interfaces{};
        // Target interface is the first one
        interfaces.push_back(extract_interface(i, j, fs, ir));
        constexpr Index NEIGHBORHOOD_SIZE = 1;
        for (Index di = -NEIGHBORHOOD_SIZE; di <= NEIGHBORHOOD_SIZE; ++di) {
          for (Index dj = -NEIGHBORHOOD_SIZE; dj <= NEIGHBORHOOD_SIZE; ++dj) {
            const bool check_here = (di != 0 || dj != 0);
            if (check_here && ir.interface.is_valid_index(i + di, j + dj) &&
                has_interface(vof, i + di, j + dj)) {
              interfaces.push_back(extract_interface(i + di, j + dj, fs, ir));
            }
          }
        }

        rotate_interfaces(interfaces);
        IGOR_ASSERT(std::abs(interfaces[0].normal[X]) < 1e-12 &&
                        std::abs(interfaces[0].normal[Y] + 1.0) < 1e-12,
                    "Expected normal of target interface to point in direction (0, -1) but is "
                    "({:.6e}, {:.6e})",
                    interfaces[0].normal[X],
                    interfaces[0].normal[Y]);

        for (const auto& interface : interfaces) {
          IGOR_ASSERT(std::abs(interface.end[X] - interface.begin[X]) > 1e-6,
                      "Interface {{ .begin = [{:.6e}, {:.6e}], .end = [{:.6e}, {:.6e}], .normal = "
                      "[{:.6e}, {:.6e}] }} is not representable by linear function.",
                      interface.begin[X],
                      interface.begin[Y],
                      interface.end[X],
                      interface.end[Y],
                      interface.normal[X],
                      interface.normal[Y]);
        }

        sort_begin_end(interfaces);
        curv[i, j] = calc_curv_quad_reconstruct(interfaces);
      } else {
        curv[i, j] = 0.0;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <size_t CAPACITY>
void solve_linear_least_squares(
    const Igor::StaticVector<std::array<Float, NDIMS>, CAPACITY>& points, Vector<Float, 3>& sol) {
  // c = (X^T * X)^(-1) * (X^T * y)

  Matrix<Float, 3, 3> A{};
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      for (const auto [xi, _] : points) {
        A[i, j] += std::pow(xi, static_cast<Float>(i)) * std::pow(xi, static_cast<Float>(j));
      }
    }
  }

  Vector<Float, 3> b{};
  for (Index i = 0; i < 3; ++i) {
    for (const auto [xi, yi] : points) {
      b[i] += std::pow(xi, static_cast<Float>(i)) * yi;
    }
  }

  solve_linear_system(A, b, sol);
}

// -------------------------------------------------------------------------------------------------
void curvature_from_quad_reconstruction2(const FS<Float, NX, NY>& fs,
                                         const InterfaceReconstruction<NX, NY>& ir,
                                         const Matrix<Float, NX, NY>& vof,
                                         Matrix<Float, NX, NY>& curv) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        Igor::StaticVector<Interface, 27UZ> interfaces{};
        // Target interface is the first one
        interfaces.push_back(extract_interface(i, j, fs, ir));
        constexpr Index NEIGHBORHOOD_SIZE = 1;
        for (Index di = -NEIGHBORHOOD_SIZE; di <= NEIGHBORHOOD_SIZE; ++di) {
          for (Index dj = -NEIGHBORHOOD_SIZE; dj <= NEIGHBORHOOD_SIZE; ++dj) {
            const bool check_here = (di != 0 || dj != 0);
            if (check_here && ir.interface.is_valid_index(i + di, j + dj) &&
                has_interface(vof, i + di, j + dj)) {
              interfaces.push_back(extract_interface(i + di, j + dj, fs, ir));
            }
          }
        }

        rotate_interfaces(interfaces);
        IGOR_ASSERT(std::abs(interfaces[0].normal[X]) < 1e-12 &&
                        std::abs(interfaces[0].normal[Y] + 1.0) < 1e-12,
                    "Expected normal of target interface to point in direction (0, -1) but is "
                    "({:.6e}, {:.6e})",
                    interfaces[0].normal[X],
                    interfaces[0].normal[Y]);

        Igor::StaticVector<std::array<Float, NDIMS>, 3UZ * 9UZ> points{};
        for (const auto& [begin, end, _] : interfaces) {
          points.push_back({
              (begin[X] + end[X]) / 2.0,
              (begin[Y] + end[Y]) / 2.0,
          });
          // points.push_back(begin);
          // points.push_back(end);
        }
        Vector<Float, 3> c{};
        solve_linear_least_squares(points, c);

        if (quad2_out != nullptr) {
          for (size_t r = 0; r < interfaces.size(); ++r) {
            const auto& interf = interfaces[r];
            (*quad2_out) << Igor::detail::format("interface {} {:.6e} {:.6e} {:.6e} {:.6e}\n",
                                                 r,
                                                 interf.begin[X],
                                                 interf.begin[Y],
                                                 interf.end[X],
                                                 interf.end[Y]);
          }
          (*quad2_out) << Igor::detail::format("quad {:.6e} {:.6e} {:.6e}\n", c[0], c[1], c[2]);
          (*quad2_out) << Igor::detail::format("eval_point {:.6e}\n", points[0][X]);
          (*quad2_out) << "------------------------------------------------------------\n";
        }

        curv[i, j] = calc_cuvature_of_quad_poly(points[0][X], c);
      } else {
        curv[i, j] = 0.0;
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------
struct CurvatureMetrics {
  Float expected_curv{};
  Float min_curv{};
  Float max_curv{};
  Float mean_curv{};
  Float mse_curv{};
  Float mrse_curv{};
};

// -------------------------------------------------------------------------------------------------
template <typename CURV_FUNC>
void test_curvature(CURV_FUNC calc_curv,
                    Float cx,
                    Float cy,
                    Float r,
                    const FS<Float, NX, NY>& fs,
                    InterfaceReconstruction<NX, NY>& ir,
                    Matrix<Float, NX, NY>& vof,
                    Matrix<Float, NX, NY>& curv,
                    CurvatureMetrics& metrics) {

  auto vof0 = [cx, cy, r](Float x, Float y) {
    return static_cast<Float>(Igor::sqr(x - cx) + Igor::sqr(y - cy) <= Igor::sqr(r));
  };
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      vof[i, j] = quadrature(vof0, fs.x[i], fs.x[i + 1], fs.y[j], fs.y[j + 1]) / (fs.dx * fs.dy);
    }
  }
  std::fill_n(ir.interface.get_data(), ir.interface.size(), IRL::PlanarSeparator{});
  reconstruct_interface(fs.x, fs.y, vof, ir);

  calc_curv(fs, ir, vof, curv);

  metrics.expected_curv = 1.0 / r;
  metrics.min_curv      = std::numeric_limits<Float>::max();
  metrics.max_curv      = -std::numeric_limits<Float>::max();
  metrics.mean_curv     = 0.0;
  metrics.mse_curv      = 0.0;
  metrics.mrse_curv     = 0.0;
  Index count           = 0;
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      if (has_interface(vof, i, j)) {
        metrics.min_curv   = std::min(curv[i, j], metrics.min_curv);
        metrics.max_curv   = std::max(curv[i, j], metrics.max_curv);
        metrics.mean_curv += curv[i, j];
        metrics.mse_curv  += Igor::sqr(curv[i, j] - metrics.expected_curv);
        metrics.mrse_curv +=
            Igor::sqr(curv[i, j] - metrics.expected_curv) / Igor::sqr(metrics.expected_curv);
        count += 1;
      }
    }
  }
  metrics.mean_curv /= static_cast<Float>(count);
  metrics.mse_curv  /= static_cast<Float>(count);
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  std::ofstream qout(Igor::detail::format("{}/quad_res.txt", OUTPUT_DIR));
  IGOR_ASSERT(
      qout, "Expected to open file `{}`.", Igor::detail::format("{}/quad_res.txt", OUTPUT_DIR));

  std::ofstream q2out(Igor::detail::format("{}/quad2_res.txt", OUTPUT_DIR));
  IGOR_ASSERT(
      q2out, "Expected to open file `{}`.", Igor::detail::format("{}/quad2_res.txt", OUTPUT_DIR));

  if (NUM_TEST_ITER == 1) {
    quad_out  = &qout;
    quad2_out = &q2out;
  }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY> fs{};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  InterfaceReconstruction<NX, NY> ir{};
  localize_cells(fs.x, fs.y, ir);

  Matrix<Float, NX, NY> vof{};
  Matrix<Float, NX, NY> smooth_curv{};

  Matrix<Float, NX, NY> quad_curv{};
  Matrix<Float, NX, NY> quad2_curv{};

  VTKWriter<Float, NX, NY> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof);
  vtk_writer.add_scalar("smooth_curv", &smooth_curv);
  vtk_writer.add_scalar("quad_curv", &quad_curv);
  vtk_writer.add_scalar("quad2_curv", &quad2_curv);

  Index iter = 0;
  Float cx   = 0.0;
  Float cy   = 0.0;
  Float r    = 0.0;
  CurvatureMetrics smooth_metrics{};
  CurvatureMetrics quad_metrics{};
  CurvatureMetrics quad2_metrics{};
  Float runtime_smooth = 0.0;
  Float runtime_quad   = 0.0;
  Float runtime_quad2  = 0.0;

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&iter, "iteration");
  monitor.add_variable(&cx, "center(x)");
  monitor.add_variable(&cy, "center(y)");
  monitor.add_variable(&r, "radius");
  monitor.add_variable(&smooth_metrics.expected_curv, "expect(curv)");

  monitor.add_variable(&smooth_metrics.min_curv, "smooth-min(curv)");
  monitor.add_variable(&smooth_metrics.max_curv, "smooth-max(curv)");
  monitor.add_variable(&smooth_metrics.mean_curv, "smooth-mean(curv)");
  monitor.add_variable(&smooth_metrics.mse_curv, "smooth-mse(curv)");
  monitor.add_variable(&smooth_metrics.mrse_curv, "smooth-mrse(curv)");
  monitor.add_variable(&runtime_smooth, "runtime_smooth [us]");

  monitor.add_variable(&quad_metrics.min_curv, "quad-min(curv)");
  monitor.add_variable(&quad_metrics.max_curv, "quad-max(curv)");
  monitor.add_variable(&quad_metrics.mean_curv, "quad-mean(curv)");
  monitor.add_variable(&quad_metrics.mse_curv, "quad-mse(curv)");
  monitor.add_variable(&quad_metrics.mrse_curv, "quad-mrse(curv)");
  monitor.add_variable(&runtime_quad, "runtime_quad [us]");

  monitor.add_variable(&quad2_metrics.min_curv, "quad2-min(curv)");
  monitor.add_variable(&quad2_metrics.max_curv, "quad2-max(curv)");
  monitor.add_variable(&quad2_metrics.mean_curv, "quad2-mean(curv)");
  monitor.add_variable(&quad2_metrics.mse_curv, "quad2-mse(curv)");
  monitor.add_variable(&quad2_metrics.mrse_curv, "quad2-mrse(curv)");
  monitor.add_variable(&runtime_quad2, "runtime_quad2 [us]");
  // = Allocate memory =============================================================================

  static std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution c_dist(0.35, 0.65);
  std::uniform_real_distribution r_dist(0.01, 0.25);

  IGOR_TIME_SCOPE("Testing cuvature") {
    Igor::ProgressBar bar(NUM_TEST_ITER, 63);
    for (iter = 0; iter < NUM_TEST_ITER; ++iter) {
      if (NUM_TEST_ITER > 1) {
        cx = c_dist(generator);
        cy = c_dist(generator);
        r  = r_dist(generator);
      } else {
        cx = 0.5;
        cy = 0.5;
        r  = 0.25;
      }

      auto t_begin = std::chrono::high_resolution_clock::now();
      test_curvature(
          calc_curvature<Float, NX, NY>, cx, cy, r, fs, ir, vof, smooth_curv, smooth_metrics);
      auto t_end     = std::chrono::high_resolution_clock::now();
      runtime_smooth = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      t_begin        = std::chrono::high_resolution_clock::now();
      test_curvature(
          curvature_from_quad_reconstruction, cx, cy, r, fs, ir, vof, quad_curv, quad_metrics);
      t_end        = std::chrono::high_resolution_clock::now();
      runtime_quad = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      t_begin      = std::chrono::high_resolution_clock::now();
      test_curvature(
          curvature_from_quad_reconstruction2, cx, cy, r, fs, ir, vof, quad2_curv, quad2_metrics);
      t_end         = std::chrono::high_resolution_clock::now();
      runtime_quad2 = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      if (!vtk_writer.write()) { return 1; }
      if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                          fs.x,
                          fs.y,
                          ir.interface)) {
        return 1;
      }
      monitor.write();
      bar.update();
    }
    std::cout << '\n';
  }

  // if (!to_npy(Igor::detail::format("{}/xm.npy", OUTPUT_DIR), fs.xm)) { return 1; }
  // if (!to_npy(Igor::detail::format("{}/ym.npy", OUTPUT_DIR), fs.ym)) { return 1; }
  // if (!to_npy(Igor::detail::format("{}/VOF.npy", OUTPUT_DIR), vof)) { return 1; }

  // for (Index i = 0; i < ir.interface.extent(0); ++i) {
  //   for (Index j = 0; j < ir.interface.extent(1); ++j) {
  //     if (has_interface(vof, i, j)) {
  //       const auto [p0, p1] =
  //           get_intersections_with_cell<Float, NX, NY>(i, j, fs.x, fs.y, ir.interface[i, j][0]);
  //       std::cout << "p0 = " << p0 << '\n';
  //       std::cout << "p1 = " << p1 << '\n';
  //       std::cout << "n  = " << ir.interface[i, j][0].normal() << '\n';
  //       std::cout << '\n';
  //     }
  //   }
  // }

  // curvature_from_quad_reconstruction(fs, ir, vof, poly_curv);
}
