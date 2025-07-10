#ifndef FLUID_SOLVER_PRESSURE_CORRECTION_FFT_HPP_
#define FLUID_SOLVER_PRESSURE_CORRECTION_FFT_HPP_

#include <poisfft.h>

#include <Igor/Math.hpp>

#include "FS.hpp"

template <typename Float, Index NX, Index NY>
class PS {
  constexpr static size_t NDIMS = 2;

  static constexpr std::array<int, NDIMS> m_n{NX, NY};
  std::array<Float, NDIMS> m_L{};
  static constexpr std::array<int, 2 * NDIMS> m_bconds{
      PoisFFT::NEUMANN, PoisFFT::NEUMANN, PoisFFT::NEUMANN, PoisFFT::NEUMANN};
  PoisFFT::Solver<NDIMS, Float> m_solver;

 public:
  // TODO: Assumes equidistant spacing in x- and y-direction respectively
  constexpr PS(Float dx, Float dy, Float /*tol*/, Index /*max_iter*/) noexcept
      : m_L({dx * NX, dy * NY}),
        m_solver(m_n.data(), m_L.data(), m_bconds.data()) {}

  // -----------------------------------------------------------------------------------------------
  constexpr PS(const PS& other) noexcept                    = delete;
  constexpr PS(PS&& other) noexcept                         = delete;
  constexpr auto operator=(const PS& other) noexcept -> PS& = delete;
  constexpr auto operator=(PS&& other) noexcept -> PS&      = delete;
  constexpr ~PS() noexcept                                  = default;

  // -------------------------------------------------------------------------------------------------
  [[nodiscard]] auto solve(const FS<Float, NX, NY>& fs,
                           const Matrix<Float, NX, NY>& div,
                           Float dt,
                           Matrix<Float, NX, NY>& resP) -> bool {
    static Matrix<Float, NX, NY, Layout::C> rhs_values{};

    // = Set right-hand side =======================================================================
    // Float mean_rhs = 0.0;
    for (Index i = 0; i < resP.extent(0); ++i) {
      for (Index j = 0; j < resP.extent(1); ++j) {
        rhs_values[i, j] = fs.rho[i, j] * div[i, j] / dt;
        // mean_rhs += rhs_values[i, j];
      }
    }
    // mean_rhs /= static_cast<Float>(rhs_values.size());
    // for (Index i = 0; i < resP.extent(0); ++i) {
    //   for (Index j = 0; j < resP.extent(1); ++j) {
    //     rhs_values[i, j] -= mean_rhs;
    //   }
    // }

    m_solver.execute(resP.get_data(), rhs_values.get_data());

    return true;
  }
};

#endif  // FLUID_SOLVER_PRESSURE_CORRECTION_FFT_HPP_
