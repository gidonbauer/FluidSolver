#include <numeric>

#include <omp.h>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "IO.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "VOF.hpp"
#include "VTKWriter.hpp"

// = Config ========================================================================================
using Float            = double;
constexpr Index NX     = 128;
constexpr Index NY     = 128;
constexpr Index NGHOST = 1;

constexpr Float X_MIN  = 0.0;
constexpr Float X_MAX  = 1.0;
constexpr Float Y_MIN  = 0.0;
constexpr Float Y_MAX  = 1.0;
constexpr auto DX      = (X_MAX - X_MIN) / static_cast<Float>(NX);
constexpr auto DY      = (Y_MAX - Y_MIN) / static_cast<Float>(NY);

constexpr Float U0     = 1.0;
constexpr Float V0     = 0.5;

Float INIT_VF_INT      = 0.0;  // NOLINT

constexpr Float DT     = 5e-3;
constexpr Index NITER  = 120;

#ifndef FS_BASE_DIR
#define FS_BASE_DIR ""
#endif  // FS_BASE_DIR
constexpr auto OUTPUT_DIR = FS_BASE_DIR "/test/output/ConstantVelocityVOF";
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
auto check_vof(const Matrix<Float, NX, NY, NGHOST>& vf) noexcept -> bool {
  const auto [min, max] = std::minmax_element(vf.get_data(), vf.get_data() + vf.size());
  const auto integral   = integrate<true>(DX, DY, vf);

  constexpr Float EPS   = 1e-12;
  if (std::abs(*min) > EPS) {
    Igor::Warn("Expected minimum VOF value to be 0 but is {:.6e}", *min);
    return false;
  }

  if (std::abs(*max - 1.0) > EPS) {
    Igor::Warn("Expected maximum VOF value to be 1 but is {:.6e}", *max);
    return false;
  }

  if (std::abs(integral - INIT_VF_INT) > EPS) {
    Igor::Warn("Expected integral of vof to be {:.6e} but is {:.6e}: error = {:.6e}",
               INIT_VF_INT,
               integral,
               std::abs(integral - INIT_VF_INT));
    return false;
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
auto calc_center_of_mass(const Vector<Float, NX, NGHOST>& xm,
                         const Vector<Float, NY, NGHOST>& ym,
                         const Matrix<Float, NX, NY, NGHOST>& vf) -> std::array<Float, 2> {
  const auto mass  = integrate(DX, DY, vf);

  Float weighted_x = 0.0;
  Float weighted_y = 0.0;
  for_each_i(vf, [&](Index i, Index j) {
    weighted_x += xm(i) * vf(i, j);
    weighted_y += ym(j) * vf(i, j);
  });
  weighted_x *= DX * DY;
  weighted_y *= DX * DY;

  return {weighted_x / mass, weighted_y / mass};
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  omp_set_num_threads(4);

  // = Create output directory =====================================================================
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{};
  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};

  VOF<Float, NX, NY, NGHOST> vof{};
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  VTKWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("VOF", &vof.vf);
  data_writer.add_vector("velocity", &Ui, &Vi);
  // = Output ======================================================================================

  // = Setup grid and cell localizers ==============================================================
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  // Localize the cells
  localize_cells(fs.x, fs.y, vof.ir);
  // = Setup grid and cell localizers ==============================================================

  // = Setup velocity and vof field ================================================================
  for_each_a<Exec::Parallel>(vof.vf, [&](Index i, Index j) {
    auto is_in = [](Float x, Float y) -> Float {
      return Igor::sqr(x - 0.25) + Igor::sqr(y - 0.25) <= Igor::sqr(0.125);
    };

    vof.vf(i, j) = quadrature(is_in, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });

  fill(fs.curr.U, U0);
  fill(fs.curr.V, V0);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  if (!data_writer.write(0.0)) { return 1; }
  INIT_VF_INT                                    = integrate<true>(fs.dx, fs.dy, vof.vf);
  const std::array<Float, 2> init_center_of_mass = calc_center_of_mass(fs.xm, fs.ym, vof.vf);
  // = Setup velocity and vof field ================================================================

  Igor::ScopeTimer timer("ConstantVelocityVOF");
  Float max_volume_error = 0.0;
  for (Index iter = 0; iter < NITER; ++iter) {
    std::copy_n(vof.vf.get_data(), vof.vf.size(), vof.vf_old.get_data());

    // = Reconstruct the interface =================================================================
    reconstruct_interface(fs, vof.vf_old, vof.ir);
    if (!save_interface(Igor::detail::format("{}/interface_{:06d}.vtk", OUTPUT_DIR, iter),
                        fs.x,
                        fs.y,
                        vof.ir.interface)) {
      return 1;
    }

    // = Advect cells ==============================================================================
    advect_cells(fs, Ui, Vi, DT, vof, &max_volume_error);

    const auto center_of_mass = calc_center_of_mass(fs.xm, fs.ym, vof.vf);
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
      if (!data_writer.write(iter + 1)) { return 1; }
    }
    if (max_volume_error > 1e-12) {
      Igor::Warn("Advected cells expanded: max. volume error = {:.6e}", max_volume_error);
      return 1;
    }
    if (!check_vof(vof.vf)) { return 1; }
  }
}
