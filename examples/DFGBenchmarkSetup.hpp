#ifndef DFG_BENCHMARK_SETUP_HPP_
#define DFG_BENCHMARK_SETUP_HPP_

#define DFG_BENCHMARK 1

#if !(DFG_BENCHMARK == 1 || DFG_BENCHMARK == 2 || DFG_BENCHMARK == 3)
#error "DFG_BENCHMARK must be 1, 2, or 3"
#endif

#include <numbers>

#include <Igor/Math.hpp>

#include "BoundaryConditions.hpp"
#include "Container.hpp"
#include "FS.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"

// = Config ========================================================================================
using Float                  = double;

constexpr Float X_MIN        = 0.0;
constexpr Float X_MAX        = 2.2;
constexpr Float Y_MIN        = 0.0;
constexpr Float Y_MAX        = 0.41;

constexpr Index NY           = 64;
constexpr Index NX           = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST       = 1;

constexpr Float T_END        = 8.0;
constexpr Float DT_MAX       = 1e-2;
constexpr Float CFL_MAX      = 0.5;
constexpr Float DT_WRITE     = 2e-2;

constexpr Float VISC         = 1e-3;
constexpr Float RHO          = 1.0;

constexpr Float CX           = 0.2;
constexpr Float CY           = 0.2;
constexpr Float R0           = 0.05;
constexpr Float L            = 2.0 * R0;
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};
constexpr auto normal_immersed_wall = [](Float x, Float y) -> std::array<Float, 2> {
  const auto d = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));
  return {
      (x - CX) / d,
      (y - CY) / d,
  };
};

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, [[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
  constexpr Float height = Y_MAX - Y_MIN;
#if DFG_BENCHMARK == 1
  const Float U = 0.3;
#elif DFG_BENCHMARK == 2
  const Float U = 1.5;
#else
  const auto U = 1.5 * std::sin(std::numbers::pi * t / 8.0);
#endif
  return (4.0 * U * y * (height - y)) / Igor::sqr(height);
}

constexpr auto U_mean([[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
#if DFG_BENCHMARK == 1
  const Float U = 0.3;
#elif DFG_BENCHMARK == 2
  const Float U = 1.5;
#else
  const auto U = 1.5 * std::sin(std::numbers::pi * t / 8.0);
#endif
  return 2.0 / 3.0 * U;
}

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
constexpr auto calc_p_diff(const FS<Float, NX, NY, NGHOST>& fs) -> Float {
  constexpr std::array<Float, 2> a1 = {0.15, 0.2};
  constexpr std::array<Float, 2> a2 = {0.25, 0.2};
  const auto p1                     = bilinear_interpolate(fs.xm, fs.ym, fs.p, a1[0], a1[1]);
  const auto p2                     = bilinear_interpolate(fs.xm, fs.ym, fs.p, a2[0], a2[1]);
  return p1 - p2;
}

constexpr auto calc_Re(Float t) -> Float { return U_mean(t) * L / VISC; }

constexpr auto calc_coefficient(auto&& f, const FS<Float, NX, NY, NGHOST>& fs, Float t) -> Float {
  const auto force = (quadrature<64>(f, 0.0, std::numbers::pi) +
                      quadrature<64>(f, std::numbers::pi, 2.0 * std::numbers::pi));

  return (2.0 * force) / (fs.rho_gas * Igor::sqr(U_mean(t)) * L);
}

constexpr auto calc_C_L(const FS<Float, NX, NY, NGHOST>& fs, Float t) {
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = CX + normal_x * R0;
    const auto y        = CY + normal_y * R0;

    const auto p        = bilinear_interpolate(fs.xm, fs.ym, fs.p, x, y);

    return -p * normal_y;
  };

  return calc_coefficient(f, fs, t);
}

constexpr auto calc_C_D(const FS<Float, NX, NY, NGHOST>& fs, Float t) {
  auto f = [&](Float theta) {
    const auto normal_x = std::cos(theta);
    const auto normal_y = std::sin(theta);
    const auto x        = CX + normal_x * R0;
    const auto y        = CY + normal_y * R0;

    const auto p        = bilinear_interpolate(fs.xm, fs.ym, fs.p, x, y);

    return -p * normal_x;
  };

  return calc_coefficient(f, fs, t);
}

#endif  // DFG_BENCHMARK_SETUP_HPP_
