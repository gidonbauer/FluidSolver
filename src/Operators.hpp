#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>
#include <Igor/MdArray.hpp>

#include "Config.hpp"
#include "FS.hpp"

// -------------------------------------------------------------------------------------------------
constexpr auto sqr(auto x) { return x * x; }

// -------------------------------------------------------------------------------------------------
void interpolate_U(const Igor::MdArray<Float, U_STAGGERED_EXTENT>& U,
                   Igor::MdArray<Float, CENTERED_EXTENT>& Ui) {
  for (size_t i = 0; i < Ui.extent(0); ++i) {
    for (size_t j = 0; j < Ui.extent(1); ++j) {
      Ui[i, j] = (U[i, j] + U[i + 1, j]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
void interpolate_V(const Igor::MdArray<Float, V_STAGGERED_EXTENT>& V,
                   Igor::MdArray<Float, CENTERED_EXTENT>& Vi) {
  for (size_t i = 0; i < Vi.extent(0); ++i) {
    for (size_t j = 0; j < Vi.extent(1); ++j) {
      Vi[i, j] = (V[i, j] + V[i, j + 1]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
void calc_divergence(const FS& fs, Igor::MdArray<Float, CENTERED_EXTENT>& div) {
  for (size_t i = 0; i < div.extent(0); ++i) {
    for (size_t j = 0; j < div.extent(1); ++j) {
      div[i, j] = (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +  //
                  (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Extent>
void calc_mid_time(Igor::MdArray<Float, Extent>& current, const Igor::MdArray<Float, Extent>& old) {
  IGOR_ASSERT(current.rank() == old.rank() && current.rank() == 2,
              "Expected rank 2 but got rank {} and {}",
              current.rank(),
              old.rank());
  IGOR_ASSERT(current.extent(0) == old.extent(0) && current.extent(1) == old.extent(1),
              "Expected same extents but got ({}, {}) and ({}, {})",
              current.extent(0),
              current.extent(1),
              old.extent(0),
              old.extent(1));

  for (size_t i = 0; i < current.extent(0); ++i) {
    for (size_t j = 0; j < current.extent(1); ++j) {
      current[i, j] = 0.5 * (current[i, j] + old[i, j]);
    }
  }
}

// -------------------------------------------------------------------------------------------------
void shift_pressure_to_zero(const FS& fs, Igor::MdArray<Float, CENTERED_EXTENT>& dp) {
  Float vol_avg_p = 0.0;

  for (size_t i = 0; i < dp.extent(0); ++i) {
    for (size_t j = 0; j < dp.extent(1); ++j) {
      vol_avg_p += dp[i, j] * fs.dx[i] * fs.dy[j];
    }
  }

  for (size_t i = 0; i < dp.extent(0); ++i) {
    for (size_t j = 0; j < dp.extent(1); ++j) {
      dp[i, j] -= vol_avg_p;
    }
  }
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
