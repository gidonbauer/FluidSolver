#ifndef FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
#define FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_

#include <variant>

#include "Container.hpp"
#include "ForEach.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
requires(NGHOST > 0)
struct FS;

// -------------------------------------------------------------------------------------------------
// TODO: Clipped Neumann?
// clang-format off
template <typename Float>
struct Dirichlet { Float U{}, V{}; };
struct Neumann {};
struct ClippedNeumann {};
struct Periodic {};
struct Symmetry {};

template <typename Float>
using BCond_t = std::variant<Dirichlet<Float>, ClippedNeumann, Neumann, Periodic, Symmetry>;
// clang-format on

template <typename Float>
struct FlowBConds {
  BCond_t<Float> left;
  BCond_t<Float> right;
  BCond_t<Float> bottom;
  BCond_t<Float> top;
};

namespace detail {

template <typename... Ts>
struct Dispatcher : Ts... {
  using Ts::operator()...;
};

}  // namespace detail

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void apply_velocity_bconds(FS<Float, NX, NY, NGHOST>& fs, const FlowBConds<Float>& bconds) {
  static_assert(NGHOST == 1, "Expected exactly one ghost cell.");

  // = Left side of domain =========================================================================
  const detail::Dispatcher left_visitor{
      [&](const Dirichlet<Float>& bcond) {
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { fs.curr.U(-NGHOST, j) = bcond.U; });
        for_each_a<Exec::Parallel>(fs.y, [&](Index j) { fs.curr.V(-NGHOST, j) = bcond.V; });
      },
      [&](const Neumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym,
                                   [&](Index j) { fs.curr.U(-NGHOST, j) = fs.curr.U(0, j); });
        for_each_a<Exec::Parallel>(fs.y, [&](Index j) { fs.curr.V(-NGHOST, j) = fs.curr.V(0, j); });
      },
      [&](const ClippedNeumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.ym, [&](Index j) { fs.curr.U(-NGHOST, j) = std::min(fs.curr.U(0, j), 0.0); });
        for_each_a<Exec::Parallel>(fs.y, [&](Index j) { fs.curr.V(-NGHOST, j) = fs.curr.V(0, j); });
      },
      [&](const Periodic& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym,
                                   [&](Index j) { fs.curr.U(-NGHOST, j) = fs.curr.U(NX - 1, j); });
        for_each_a<Exec::Parallel>(fs.y,
                                   [&](Index j) { fs.curr.V(-NGHOST, j) = fs.curr.V(NX - 1, j); });
      },
      [&](const Symmetry& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          fs.curr.U(-NGHOST, j) = -fs.curr.U(1, j);
          fs.curr.U(0, j)       = 0.0;
        });
        for_each_a<Exec::Parallel>(fs.y, [&](Index j) { fs.curr.V(-NGHOST, j) = fs.curr.V(0, j); });
      },
  };
  std::visit(left_visitor, bconds.left);

  // = Right side of domain ========================================================================
  const detail::Dispatcher right_visitor{
      [&](const Dirichlet<Float>& bcond) {
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) { fs.curr.U(NX + NGHOST, j) = bcond.U; });
        for_each_a<Exec::Parallel>(fs.y, [&](Index j) { fs.curr.V(NX + NGHOST - 1, j) = bcond.V; });
      },
      [&](const Neumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym,
                                   [&](Index j) { fs.curr.U(NX + NGHOST, j) = fs.curr.U(NX, j); });
        for_each_a<Exec::Parallel>(
            fs.y, [&](Index j) { fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(NX - 1, j); });
      },
      [&](const ClippedNeumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.ym, [&](Index j) { fs.curr.U(NX + NGHOST, j) = std::max(fs.curr.U(NX, j), 0.0); });
        for_each_a<Exec::Parallel>(
            fs.y, [&](Index j) { fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(NX - 1, j); });
      },
      [&](const Periodic& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym,
                                   [&](Index j) { fs.curr.U(NX + NGHOST, j) = fs.curr.U(1, j); });
        for_each_a<Exec::Parallel>(
            fs.y, [&](Index j) { fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(0, j); });
      },
      [&](const Symmetry& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
          fs.curr.U(NX + NGHOST, j) = -fs.curr.U(NX - 1, j);
          fs.curr.U(NX, j)          = 0.0;
        });
        for_each_a<Exec::Parallel>(
            fs.y, [&](Index j) { fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(NX - 1, j); });
      },
  };
  std::visit(right_visitor, bconds.right);

  // = Bottom side of domain =======================================================================
  const detail::Dispatcher bottom_visitor{
      [&](const Dirichlet<Float>& bcond) {
        for_each_a<Exec::Parallel>(fs.x, [&](Index i) { fs.curr.U(i, -NGHOST) = bcond.U; });
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { fs.curr.V(i, -NGHOST) = bcond.V; });
      },
      [&](const Neumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.x, [&](Index i) { fs.curr.U(i, -NGHOST) = fs.curr.U(i, 0); });
        for_each_a<Exec::Parallel>(fs.xm,
                                   [&](Index i) { fs.curr.V(i, -NGHOST) = fs.curr.V(i, 0); });
      },
      [&](const ClippedNeumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.x, [&](Index i) { fs.curr.U(i, -NGHOST) = fs.curr.U(i, 0); });
        for_each_a<Exec::Parallel>(
            fs.xm, [&](Index i) { fs.curr.V(i, -NGHOST) = std::min(fs.curr.V(i, 0), 0.0); });
      },
      [&](const Periodic& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.x,
                                   [&](Index i) { fs.curr.U(i, -NGHOST) = fs.curr.U(i, NY - 1); });
        for_each_a<Exec::Parallel>(fs.xm,
                                   [&](Index i) { fs.curr.V(i, -NGHOST) = fs.curr.V(i, NY - 1); });
      },
      [&](const Symmetry& /*bcond*/) {
        for_each_a<Exec::Parallel>(fs.x, [&](Index i) { fs.curr.U(i, -NGHOST) = fs.curr.U(i, 0); });
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          fs.curr.V(i, -NGHOST) = -fs.curr.V(i, 1);
          fs.curr.V(i, 0)       = 0.0;
        });
      },
  };
  std::visit(bottom_visitor, bconds.bottom);

  // = Top side of domain ==========================================================================
  const detail::Dispatcher top_visitor{
      [&](const Dirichlet<Float>& bcond) {
        for_each_a<Exec::Parallel>(fs.x, [&](Index i) { fs.curr.U(i, NY + NGHOST - 1) = bcond.U; });
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) { fs.curr.V(i, NY + NGHOST) = bcond.V; });
      },
      [&](const Neumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.x, [&](Index i) { fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, NY - 1); });
        for_each_a<Exec::Parallel>(fs.xm,
                                   [&](Index i) { fs.curr.V(i, NY + NGHOST) = fs.curr.V(i, NY); });
      },
      [&](const ClippedNeumann& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.x, [&](Index i) { fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, NY - 1); });
        for_each_a<Exec::Parallel>(
            fs.xm, [&](Index i) { fs.curr.V(i, NY + NGHOST) = std::max(fs.curr.V(i, NY), 0.0); });
      },
      [&](const Periodic& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.x, [&](Index i) { fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, 0); });
        for_each_a<Exec::Parallel>(fs.xm,
                                   [&](Index i) { fs.curr.V(i, NY + NGHOST) = fs.curr.V(i, 1); });
      },
      [&](const Symmetry& /*bcond*/) {
        for_each_a<Exec::Parallel>(
            fs.x, [&](Index i) { fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, NY - 1); });
        for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
          fs.curr.V(i, NY + NGHOST) = -fs.curr.V(i, NY - 1);
          fs.curr.V(i, NY)          = 0.0;
        });
      },
  };
  std::visit(top_visitor, bconds.top);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void apply_neumann_bconds(Matrix<Float, NX, NY, NGHOST>& field) noexcept {
  static_assert(NGHOST > 0, "Expected at least one ghost cell.");

  for_each<-NGHOST, NY + NGHOST, Exec::Parallel>([&](Index j) {
    // LEFT
    for (Index i = -NGHOST; i < 0; ++i) {
      field(i, j) = field(0, j);
    }
    // RIGHT
    for (Index i = NX; i < NX + NGHOST; ++i) {
      field(i, j) = field(NX - 1, j);
    }
  });

  for_each<-NGHOST, NX + NGHOST, Exec::Parallel>([&](Index i) {
    // BOTTOM
    for (Index j = -NGHOST; j < 0; ++j) {
      field(i, j) = field(i, 0);
    }
    // TOP
    for (Index j = NY; j < NY + NGHOST; ++j) {
      field(i, j) = field(i, NY - 1);
    }
  });
}

#endif  // FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
