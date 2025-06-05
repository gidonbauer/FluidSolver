#ifndef FLUID_SOLVER_CONFIG_HPP_
#define FLUID_SOLVER_CONFIG_HPP_

#include <cstddef>
#include <mdspan>

#include <Igor/MdArray.hpp>

using Float = double;

constexpr size_t NDIMS = 2;
constexpr size_t NX    = 10;  // 500;
constexpr size_t NY    = 10;  // 400;

constexpr Float X_MIN = 0.0;
constexpr Float X_MAX = 1.0;
constexpr Float Y_MIN = 0.0;
constexpr Float Y_MAX = 1.0;

constexpr Float T_END   = 0.5;
constexpr Float DT_MAX  = 1e-3;
constexpr Float CFL_MAX = 0.5;

constexpr Float U_IN = 2.0;
constexpr Float VISC = 1e-3;
constexpr Float RHO  = 0.9;

constexpr size_t NUM_SUBITER = 2;

constexpr auto OUTPUT_DIR = "output";

using NX_EXTENT          = std::extents<size_t, NX>;
using NX_P1_EXTENT       = std::extents<size_t, NX + 1>;
using NY_EXTENT          = std::extents<size_t, NY>;
using NY_P1_EXTENT       = std::extents<size_t, NY + 1>;
using CENTERED_EXTENT    = std::extents<size_t, NX, NY>;
using U_STAGGERED_EXTENT = std::extents<size_t, NX + 1, NY>;
using V_STAGGERED_EXTENT = std::extents<size_t, NX, NY + 1>;

[[nodiscard]] constexpr auto make_nx() noexcept { return Igor::MdArray<Float, NX_EXTENT>(NX); }
[[nodiscard]] constexpr auto make_nx_p1() noexcept {
  return Igor::MdArray<Float, NX_P1_EXTENT>(NX + 1);
}

[[nodiscard]] constexpr auto make_ny() noexcept { return Igor::MdArray<Float, NY_EXTENT>(NY); }
[[nodiscard]] constexpr auto make_ny_p1() noexcept {
  return Igor::MdArray<Float, NY_P1_EXTENT>(NY + 1);
}

[[nodiscard]] constexpr auto make_centered() noexcept {
  return Igor::MdArray<Float, CENTERED_EXTENT>(NX, NY);
}

[[nodiscard]] constexpr auto make_u_staggered() noexcept {
  return Igor::MdArray<Float, U_STAGGERED_EXTENT>(NX + 1, NY);
}
[[nodiscard]] constexpr auto make_v_staggered() noexcept {
  return Igor::MdArray<Float, V_STAGGERED_EXTENT>(NX, NY + 1);
}

#endif  // FLUID_SOLVER_CONFIG_HPP_
