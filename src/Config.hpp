#ifndef FLUID_SOLVER_CONFIG_HPP_
#define FLUID_SOLVER_CONFIG_HPP_

#include <cstddef>

#include "Container.hpp"

using Float = double;

constexpr size_t NDIMS = 2;
constexpr size_t NX    = 1024;  // 5;
constexpr size_t NY    = 256;   // 5;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 100.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr Float T_END    = 20.0;
constexpr Float DT_MAX   = 1e-1;
constexpr Float CFL_MAX  = 0.9;
constexpr Float DT_WRITE = 0.5;

constexpr Float U_IN  = 1.0;
constexpr Float U_TOP = 0.0;
constexpr Float U_BOT = 0.0;
constexpr Float VISC  = 1e-3;
constexpr Float RHO   = 0.9;

constexpr int PRESSURE_MAX_ITER = 500;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr size_t NUM_SUBITER = 2;

constexpr auto OUTPUT_DIR = "output";

[[nodiscard]] constexpr auto make_nx() noexcept { return Vector<Float, NX>(); }
[[nodiscard]] constexpr auto make_nx_p1() noexcept { return Vector<Float, NX + 1>(); }
[[nodiscard]] constexpr auto make_ny() noexcept { return Vector<Float, NY>(); }
[[nodiscard]] constexpr auto make_ny_p1() noexcept { return Vector<Float, NY + 1>(); }
[[nodiscard]] constexpr auto make_centered() noexcept { return Matrix<Float, NX, NY>(); }
[[nodiscard]] constexpr auto make_u_staggered() noexcept { return Matrix<Float, NX + 1, NY>(); }
[[nodiscard]] constexpr auto make_v_staggered() noexcept { return Matrix<Float, NX, NY + 1>(); }

#endif  // FLUID_SOLVER_CONFIG_HPP_
