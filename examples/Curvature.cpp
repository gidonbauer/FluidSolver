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
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float                   = double;
constexpr Index N             = 64;
constexpr Index NGHOST        = 1;

constexpr Float X_MIN         = 0.0;
constexpr Float X_MAX         = 1.0;
constexpr Float Y_MIN         = 0.0;
constexpr Float Y_MAX         = 1.0;
constexpr Float DX            = (X_MAX - X_MIN) / static_cast<Float>(N);
constexpr Float DY            = (Y_MAX - Y_MIN) / static_cast<Float>(N);

constexpr Index NUM_TEST_ITER = 50'000;
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
struct CurvatureMetrics {
  Float expected_curv{};
  Float min_curv{};
  Float max_curv{};
  Float mean_curv{};
  Float mse_curv{};
  Float mrse_curv{};
  Float init_error{};
};

// -------------------------------------------------------------------------------------------------
template <typename CURV_FUNC>
void test_curvature(CURV_FUNC calc_curv,
                    Float cx,
                    Float cy,
                    Float r,
                    bool invert_phases,
                    const FS<Float, N, N, NGHOST>& fs,
                    VOF<Float, N, N, NGHOST>& vof,
                    Matrix<Float, N, N, NGHOST>& curv,
                    CurvatureMetrics& metrics) {

  auto vof0 = [cx, cy, r, invert_phases](Float x, Float y) {
    if (invert_phases) {
      return static_cast<Float>(Igor::sqr(x - cx) + Igor::sqr(y - cy) > Igor::sqr(r));
    }
    return static_cast<Float>(Igor::sqr(x - cx) + Igor::sqr(y - cy) <= Igor::sqr(r));
  };

  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    vof.vf_old(i, j) =
        quadrature<64>(vof0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
    vof.vf(i, j) = vof.vf_old(i, j);
  });
  std::fill_n(vof.ir.interface.get_data(), vof.ir.interface.size(), IRL::PlanarSeparator{});
  reconstruct_interface(fs, vof.vf, vof.ir);

  if (invert_phases) {
    const auto domain_area = (X_MAX - X_MIN) * (Y_MAX - Y_MIN);
    metrics.init_error     = std::abs((domain_area - integrate(fs.dx, fs.dy, vof.vf)) -
                                  std::numbers::pi * Igor::sqr(r)) /
                         (std::numbers::pi * Igor::sqr(r));
  } else {
    metrics.init_error =
        std::abs(integrate(fs.dx, fs.dy, vof.vf) - std::numbers::pi * Igor::sqr(r)) /
        (std::numbers::pi * Igor::sqr(r));
  }

  calc_curv(fs, vof);
  std::copy_n(vof.curv.get_data(), vof.curv.size(), curv.get_data());

  metrics.expected_curv = 1.0 / r * (invert_phases ? -1.0 : 1.0);
  metrics.min_curv      = std::numeric_limits<Float>::max();
  metrics.max_curv      = -std::numeric_limits<Float>::max();
  metrics.mean_curv     = 0.0;
  metrics.mse_curv      = 0.0;
  metrics.mrse_curv     = 0.0;
  Index count           = 0;
  for_each_i(curv, [&](Index i, Index j) {
    if (has_interface(vof.vf, i, j)) {
      metrics.min_curv   = std::min(curv(i, j), metrics.min_curv);
      metrics.max_curv   = std::max(curv(i, j), metrics.max_curv);
      metrics.mean_curv += curv(i, j);
      metrics.mse_curv  += Igor::sqr(curv(i, j) - metrics.expected_curv);
      metrics.mrse_curv +=
          Igor::sqr(curv(i, j) - metrics.expected_curv) / Igor::sqr(metrics.expected_curv);
      count += 1;
    }
  });
  metrics.mean_curv /= static_cast<Float>(count);
  metrics.mse_curv  /= static_cast<Float>(count);
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, N, N, NGHOST> fs{};
  init_grid(X_MIN, X_MAX, N, Y_MIN, Y_MAX, N, fs);

  VOF<Float, N, N, NGHOST> vof{};
  localize_cells(fs.x, fs.y, vof.ir);

  Matrix<Float, N, N, NGHOST> curv_cv{};
  Matrix<Float, N, N, NGHOST> curv_quad_vol_match{};
  Matrix<Float, N, N, NGHOST> curv_quad_regression{};

  VTKWriter<Float, N, N, NGHOST> vtk_writer(OUTPUT_DIR, &fs.x, &fs.y);
  vtk_writer.add_scalar("VOF", &vof.vf);
  vtk_writer.add_scalar("curv_cv", &curv_cv);
  vtk_writer.add_scalar("curv_quad_vol_match", &curv_quad_vol_match);
  vtk_writer.add_scalar("curv_quad_regression", &curv_quad_regression);

  Index iter        = 0;
  Float cx          = 0.0;
  Float cy          = 0.0;
  Float r           = 0.0;
  Float cells_per_r = 0.0;
  Index invert      = 0;
  CurvatureMetrics metrics_cv{};
  CurvatureMetrics metrics_quad_vol_match{};
  CurvatureMetrics metrics_quad_regression{};
  Float runtime_cv              = 0.0;
  Float runtime_quad_vol_match  = 0.0;
  Float runtime_quad_regression = 0.0;

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&iter, "iteration");
  monitor.add_variable(&cx, "center-x");
  monitor.add_variable(&cy, "center-y");
  monitor.add_variable(&r, "radius");
  monitor.add_variable(&cells_per_r, "cells-per-radius");
  monitor.add_variable(&invert, "invert");
  monitor.add_variable(&metrics_cv.expected_curv, "expect(curv)");
  monitor.add_variable(&metrics_cv.init_error, "init. error");

  monitor.add_variable(&metrics_cv.min_curv, "cv-min(curv)");
  monitor.add_variable(&metrics_cv.max_curv, "cv-max(curv)");
  monitor.add_variable(&metrics_cv.mean_curv, "cv-mean(curv)");
  monitor.add_variable(&metrics_cv.mse_curv, "cv-mse(curv)");
  monitor.add_variable(&metrics_cv.mrse_curv, "cv-mrse(curv)");
  monitor.add_variable(&runtime_cv, "cv-runtime [us]");

  monitor.add_variable(&metrics_quad_vol_match.min_curv, "quad-vol-min(curv)");
  monitor.add_variable(&metrics_quad_vol_match.max_curv, "quad-vol-max(curv)");
  monitor.add_variable(&metrics_quad_vol_match.mean_curv, "quad-vol-mean(curv)");
  monitor.add_variable(&metrics_quad_vol_match.mse_curv, "quad-vol-mse(curv)");
  monitor.add_variable(&metrics_quad_vol_match.mrse_curv, "quad-vol-mrse(curv)");
  monitor.add_variable(&runtime_quad_vol_match, "quad-vol-runtime [us]");

  monitor.add_variable(&metrics_quad_regression.min_curv, "quad-reg-min(curv)");
  monitor.add_variable(&metrics_quad_regression.max_curv, "quad-reg-max(curv)");
  monitor.add_variable(&metrics_quad_regression.mean_curv, "quad-reg-mean(curv)");
  monitor.add_variable(&metrics_quad_regression.mse_curv, "quad-reg-mse(curv)");
  monitor.add_variable(&metrics_quad_regression.mrse_curv, "quad-reg-mrse(curv)");
  monitor.add_variable(&runtime_quad_regression, "quad-reg-runtime [us]");
  // = Allocate memory =============================================================================

  static std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution c_dist(0.35, 0.65);
  std::uniform_real_distribution r_dist(2 * std::min(DX, DY), 20 * std::min(DX, DY));
  std::uniform_int_distribution<Index> i_dist(0, 1);

  IGOR_TIME_SCOPE("Testing cuvature") {
    Igor::ProgressBar bar(NUM_TEST_ITER, 63);
    for (iter = 0; iter < NUM_TEST_ITER; ++iter) {
      cx     = c_dist(generator);
      cy     = c_dist(generator);
      r      = r_dist(generator);
      invert = i_dist(generator);

      while ((cx - (r + 2 * DX) < X_MIN) || (cx + (r + 2 * DX) > X_MAX) ||
             (cy - (r + 2 * DY) < Y_MIN) || (cy + (r + 2 * DY) > Y_MAX)) {
        r /= 2.0;
      }

      auto t_begin = std::chrono::high_resolution_clock::now();
      test_curvature(calc_curvature_convolved_vf<Float, N, N, NGHOST>,
                     cx,
                     cy,
                     r,
                     invert == 1,
                     fs,
                     vof,
                     curv_cv,
                     metrics_cv);
      auto t_end = std::chrono::high_resolution_clock::now();
      runtime_cv = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      t_begin    = std::chrono::high_resolution_clock::now();
      test_curvature(calc_curvature_quad_volume_matching<Float, N, N, NGHOST>,
                     cx,
                     cy,
                     r,
                     invert == 1,
                     fs,
                     vof,
                     curv_quad_vol_match,
                     metrics_quad_vol_match);
      t_end                  = std::chrono::high_resolution_clock::now();
      runtime_quad_vol_match = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      t_begin                = std::chrono::high_resolution_clock::now();
      test_curvature(calc_curvature_quad_regression<Float, N, N, NGHOST>,
                     cx,
                     cy,
                     r,
                     invert == 1,
                     fs,
                     vof,
                     curv_quad_regression,
                     metrics_quad_regression);
      t_end                   = std::chrono::high_resolution_clock::now();
      runtime_quad_regression = std::chrono::duration<double, std::micro>(t_end - t_begin).count();

      if (invert == 1) { r *= -1.0; }
      cells_per_r = r * N;

      if (!vtk_writer.write()) { return 1; }
      if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                          fs.x,
                          fs.y,
                          vof.ir.interface)) {
        return 1;
      }
      monitor.write();
      bar.update();
    }
    std::cout << '\n';
  }
}
