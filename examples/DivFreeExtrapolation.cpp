#include <numbers>

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "ForEach.hpp"
#include "IO.hpp"
#include "Operators.hpp"

#if defined(USE_VTK) || defined(FS_DISABLE_HDF)
#include "VTKWriter.hpp"
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = VTKWriter<Float, NX, NY, NGHOST>;
#else
#include "XDMFWriter.hpp"
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = XDMFWriter<Float, NX, NY, NGHOST>;
#endif  // USE_VTK

// = Setup =========================================================================================
using Float            = double;
constexpr Index NX     = 16;
constexpr Index NY     = 16;
constexpr Index NGHOST = 1;

constexpr Float X_MIN  = 0.0;
constexpr Float X_MAX  = 1.0;
constexpr Float Y_MIN  = 0.0;
constexpr Float Y_MAX  = 1.0;

constexpr Float VISC   = 1.0;
constexpr Float RHO    = 1.0;

#ifndef FS_BASE_DIR
#define FS_BASE_DIR "."
#endif  // FS_BASE_DIR
constexpr auto OUTPUT_DIR = FS_BASE_DIR "/output/DivFreeExtrapolation/";
// = Setup =========================================================================================

// =================================================================================================
auto F(Float t) -> Float { return std::exp(-2.0 * VISC / RHO * t); }
auto u_analytical(Float x, Float y, Float t) -> Float {
  return std::sin(2.0 * std::numbers::pi * x) * std::cos(2.0 * std::numbers::pi * y) * F(t);
}
auto v_analytical(Float x, Float y, Float t) -> Float {
  return -std::cos(2.0 * std::numbers::pi * x) * std::sin(2.0 * std::numbers::pi * y) * F(t);
}

// =================================================================================================
auto main() -> int {
  if (!init_output_directory(OUTPUT_DIR)) {
    Igor::Warn("Could not initialize the output directory `{}`", OUTPUT_DIR);
    return 1;
  }

  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};
  Matrix<Float, NX, NY, NGHOST> ext{};

  // for_each<NX / 4, 3 * NX / 4 + 1, NY / 4, 3 * NY / 4>(
  //     [&](Index i, Index j) { fs.curr.U(i, j) = u_analytical(fs.x(i), fs.ym(j), 0.0); });
  // for_each<NX / 4, 3 * NX / 4, NY / 4, 3 * NY / 4 + 1>(
  //     [&](Index i, Index j) { fs.curr.V(i, j) = v_analytical(fs.xm(i), fs.y(j), 0.0); });

  for_each_i(fs.curr.U, [&](Index i, Index j) {
    const auto x = fs.x(i);
    const auto y = fs.ym(j);
    if (Igor::sqr(x - 0.5) + Igor::sqr(y - 0.5) <= Igor::sqr(0.25)) {
      fs.curr.U(i, j) = u_analytical(x, y, 0.0);
    }
  });
  for_each_i(fs.curr.V, [&](Index i, Index j) {
    const auto x = fs.xm(i);
    const auto y = fs.y(j);
    if (Igor::sqr(x - 0.5) + Igor::sqr(y - 0.5) <= Igor::sqr(0.25)) {
      fs.curr.V(i, j) = v_analytical(x, y, 0.0);
    }
  });

  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("div", &div);
  data_writer.add_scalar("ext", &ext);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  for_each_i(ext,
             [&](Index i, Index j) { ext(i, j) = static_cast<Float>(std::abs(div(i, j)) > 1e-8); });

  if (!to_npy(Igor::detail::format("{}/x.npy", OUTPUT_DIR), fs.x)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/xm.npy", OUTPUT_DIR), fs.xm)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/y.npy", OUTPUT_DIR), fs.y)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/ym.npy", OUTPUT_DIR), fs.ym)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/U.npy", OUTPUT_DIR), fs.curr.U)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/V.npy", OUTPUT_DIR), fs.curr.V)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/div.npy", OUTPUT_DIR), div)) { return 1; }
  if (!to_npy(Igor::detail::format("{}/ext.npy", OUTPUT_DIR), ext)) { return 1; }
  if (!data_writer.write(0.0)) { return 1; }

  Igor::Warn("TODO: Implement the divergence-free extrapolation of the velocity field.");
}
