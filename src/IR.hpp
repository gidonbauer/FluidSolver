#ifndef FLUID_SOLVER_IR_HPP_
#define FLUID_SOLVER_IR_HPP_

// Disable warnings for IRL
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wnullability-extension"
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include <irl/geometry/general/pt.h>
#include <irl/interface_reconstruction_methods/elvira_neighborhood.h>
#include <irl/interface_reconstruction_methods/reconstruction_interface.h>
#pragma clang diagnostic pop

#include "Container.hpp"

template <Index NX, Index NY>
struct InterfaceReconstruction {
  Matrix<IRL::PlanarSeparator, NX, NY> interface;
  Matrix<IRL::PlanarLocalizer, NX, NY> cell_localizer;
};

inline constexpr double VF_LOW  = 1e-8;
inline constexpr double VF_HIGH = 1.0 - VF_LOW;

template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto has_interface(const Matrix<Float, NX, NY, NGHOST>& vf, Index i, Index j) noexcept
    -> bool {
  return VF_LOW < vf[i, j] && vf[i, j] < VF_HIGH;
}

template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto has_interface_in_neighborhood(const Matrix<Float, NX, NY, NGHOST>& vf,
                                             Index i,
                                             Index j,
                                             Index neighborhood_size) noexcept -> bool {
  for (Index di = -neighborhood_size; di <= neighborhood_size; ++di) {
    for (Index dj = -neighborhood_size; dj <= neighborhood_size; ++dj) {
      if (vf.is_valid_index(i + di, j + dj) && has_interface(vf, i + di, j + dj)) { return true; }
    }
  }
  return false;
}

#endif  // FLUID_SOLVER_IR_HPP_
