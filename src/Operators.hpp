#ifndef FLUID_SOLVER_OPERATORS_HPP_
#define FLUID_SOLVER_OPERATORS_HPP_

#include <Igor/Logging.hpp>

#include "FS.hpp"

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void interpolate_U(const Matrix<Float, NX + 1, NY>& U, Matrix<Float, NX, NY>& Ui) {
  for (Index i = 0; i < Ui.extent(0); ++i) {
    for (Index j = 0; j < Ui.extent(1); ++j) {
      Ui[i, j] = (U[i, j] + U[i + 1, j]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void interpolate_V(const Matrix<Float, NX, NY + 1>& V, Matrix<Float, NX, NY>& Vi) {
  for (Index i = 0; i < Vi.extent(0); ++i) {
    for (Index j = 0; j < Vi.extent(1); ++j) {
      Vi[i, j] = (V[i, j] + V[i, j + 1]) / 2;
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_divergence(const FS<Float, NX, NY>& fs, Matrix<Float, NX, NY>& div) {
  for (Index i = 0; i < div.extent(0); ++i) {
    for (Index j = 0; j < div.extent(1); ++j) {
      div[i, j] = (fs.U[i + 1, j] - fs.U[i, j]) / fs.dx[i] +  //
                  (fs.V[i, j + 1] - fs.V[i, j]) / fs.dy[j];
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void calc_mid_time(Matrix<Float, NX, NY>& current, const Matrix<Float, NX, NY>& old) {
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      current[i, j] = 0.5 * (current[i, j] + old[i, j]);
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void shift_pressure_to_zero(const FS<Float, NX, NY>& fs, Matrix<Float, NX, NY>& dp) {
  Float vol_avg_p = 0.0;

  for (Index i = 0; i < dp.extent(0); ++i) {
    for (Index j = 0; j < dp.extent(1); ++j) {
      vol_avg_p += dp[i, j] * fs.dx[i] * fs.dy[j];
    }
  }

  for (Index i = 0; i < dp.extent(0); ++i) {
    for (Index j = 0; j < dp.extent(1); ++j) {
      dp[i, j] -= vol_avg_p;
    }
  }
}

#endif  // FLUID_SOLVER_OPERATORS_HPP_
