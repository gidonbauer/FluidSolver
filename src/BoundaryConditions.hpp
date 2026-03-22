#ifndef FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
#define FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_

#include <variant>

#include "Container.hpp"
#include "Macros.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
requires(NGHOST > 0)
struct FS;

// -------------------------------------------------------------------------------------------------
template <typename Float>
struct Dirichlet {
  std::variant<Float, Float (*)(Float, Float)> U{}, V{};

  // = LEFT ========
  template <Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_left(FS<Float, NX, NY, 1>& fs, Float t) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) {
      const Float U_bc =
          std::holds_alternative<Float>(U) ? std::get<0>(U) : std::get<1>(U)(fs.ym(j), t);
      fs.curr.U(-1, j) = U_bc;
      fs.curr.U(0, j)  = U_bc;
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) {
      const Float V_bc =
          std::holds_alternative<Float>(V) ? std::get<0>(V) : std::get<1>(V)(fs.y(j), t);
      fs.curr.V(-1, j) = 2.0 * V_bc - fs.curr.V(0, j);
    }
  }

  // = RIGHT =======
  template <Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_right(FS<Float, NX, NY, 1>& fs, Float t) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) {
      const Float U_bc =
          std::holds_alternative<Float>(U) ? std::get<0>(U) : std::get<1>(U)(fs.ym(j), t);
      fs.curr.U(NX, j)     = U_bc;
      fs.curr.U(NX + 1, j) = U_bc;
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) {
      const Float V_bc =
          std::holds_alternative<Float>(V) ? std::get<0>(V) : std::get<1>(V)(fs.y(j), t);
      fs.curr.V(NX, j) = 2.0 * V_bc - fs.curr.V(NX - 1, j);
    }
  }

  // = BOTTOM ======
  template <Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_bottom(FS<Float, NX, NY, 1>& fs, Float t) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) {
      const Float U_bc =
          std::holds_alternative<Float>(U) ? std::get<0>(U) : std::get<1>(U)(fs.x(i), t);
      fs.curr.U(i, -1) = 2.0 * U_bc - fs.curr.U(i, 0);
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) {
      const Float V_bc =
          std::holds_alternative<Float>(V) ? std::get<0>(V) : std::get<1>(V)(fs.xm(i), t);
      fs.curr.V(i, -1) = V_bc;
      fs.curr.V(i, 0)  = V_bc;
    }
  }

  // = TOP =========
  template <Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_top(FS<Float, NX, NY, 1>& fs, Float t) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) {
      const Float U_bc =
          std::holds_alternative<Float>(U) ? std::get<0>(U) : std::get<1>(U)(fs.x(i), t);
      fs.curr.U(i, NY) = 2.0 * U_bc - fs.curr.U(i, NY - 1);
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) {
      const Float V_bc =
          std::holds_alternative<Float>(V) ? std::get<0>(V) : std::get<1>(V)(fs.xm(i), t);
      fs.curr.V(i, NY)     = V_bc;
      fs.curr.V(i, NY + 1) = V_bc;
    }
  }
};

// -------------------------------------------------------------------------------------------------
struct Neumann {
  bool clipped = false;

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_left(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    if (clipped) {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(-1, j) = std::min(fs.curr.U(0, j), 0.0); }
    } else {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(-1, j) = fs.curr.U(0, j); }
    }
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(-1, j) = fs.curr.V(0, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_right(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    if (clipped) {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(NX + 1, j) = std::max(fs.curr.U(NX, j), 0.0); }
    } else {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(NX + 1, j) = fs.curr.U(NX, j); }
    }
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(NX, j) = fs.curr.V(NX - 1, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_bottom(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, -1) = fs.curr.U(i, 0); }
    if (clipped) {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, -1) = std::min(fs.curr.V(i, 0), 0.0); }
    } else {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, -1) = fs.curr.V(i, 0); }
    }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_top(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, NY) = fs.curr.U(i, NY - 1); }
    if (clipped) {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, NY + 1) = std::max(fs.curr.V(i, NY), 0.0); }
    } else {
      FS_PARALLEL_FOR()
      FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, NY + 1) = fs.curr.V(i, NY); }
    }
  }
};

// -------------------------------------------------------------------------------------------------
struct Periodic {
  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_left(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(-1, j) = fs.curr.U(NX - 1, j); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(-1, j) = fs.curr.V(NX - 1, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_right(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) { fs.curr.U(NX + 1, j) = fs.curr.U(1, j); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(NX, j) = fs.curr.V(0, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_bottom(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, -1) = fs.curr.U(i, NY - 1); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, -1) = fs.curr.V(i, NY - 1); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_top(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, NY) = fs.curr.U(i, 0); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) { fs.curr.V(i, NY + 1) = fs.curr.V(i, 1); }
  }
};

// -------------------------------------------------------------------------------------------------
struct Symmetry {
  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_left(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) {
      fs.curr.U(-1, j) = -fs.curr.U(1, j);
      fs.curr.U(0, j)  = 0.0;
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(-1, j) = fs.curr.V(0, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_right(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.ym, j) {
      fs.curr.U(NX + 1, j) = -fs.curr.U(NX - 1, j);
      fs.curr.U(NX, j)     = 0.0;
    }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.y, j) { fs.curr.V(NX, j) = fs.curr.V(NX - 1, j); }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_bottom(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, -1) = fs.curr.U(i, 0); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) {
      fs.curr.V(i, -1) = -fs.curr.V(i, 1);
      fs.curr.V(i, 0)  = 0.0;
    }
  }

  template <typename Float, Index NX, Index NY>
  FS_PARALLEL_CONSTEXPR void apply_top(FS<Float, NX, NY, 1>& fs, Float /*t*/) const noexcept {
    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.x, i) { fs.curr.U(i, NY) = fs.curr.U(i, NY - 1); }

    FS_PARALLEL_FOR()
    FS_FOR_EACH_A_1D(fs.xm, i) {
      fs.curr.V(i, NY + 1) = -fs.curr.V(i, NY - 1);
      fs.curr.V(i, NY)     = 0.0;
    }
  }
};

// -------------------------------------------------------------------------------------------------
template <typename Float>
using BCond_t = std::variant<Dirichlet<Float>, Neumann, Periodic, Symmetry>;

template <typename Float>
struct FlowBConds {
  BCond_t<Float> left;
  BCond_t<Float> right;
  BCond_t<Float> bottom;
  BCond_t<Float> top;
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void apply_velocity_bconds(FS<Float, NX, NY, NGHOST>& fs,
                           const FlowBConds<Float>& bconds,
                           Float t = -1.0) {
  static_assert(NGHOST == 1, "Expected exactly one ghost cell.");
  std::visit([&](auto&& bcond) { bcond.apply_left(fs, t); }, bconds.left);
  std::visit([&](auto&& bcond) { bcond.apply_right(fs, t); }, bconds.right);
  std::visit([&](auto&& bcond) { bcond.apply_bottom(fs, t); }, bconds.bottom);
  std::visit([&](auto&& bcond) { bcond.apply_top(fs, t); }, bconds.top);
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
FS_PARALLEL_CONSTEXPR void apply_neumann_bconds(Field2D<Float, NX, NY, NGHOST>& field) noexcept {
  static_assert(NGHOST > 0, "Expected at least one ghost cell.");

  FS_PARALLEL_FOR()
  for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
    // LEFT
    for (Index i = -NGHOST; i < 0; ++i) {
      field(i, j) = field(0, j);
    }
    // RIGHT
    for (Index i = NX; i < NX + NGHOST; ++i) {
      field(i, j) = field(NX - 1, j);
    }
  }

  FS_PARALLEL_FOR()
  for (Index i = -NGHOST; i < NX + NGHOST; ++i) {
    // BOTTOM
    for (Index j = -NGHOST; j < 0; ++j) {
      field(i, j) = field(i, 0);
    }
    // TOP
    for (Index j = NY; j < NY + NGHOST; ++j) {
      field(i, j) = field(i, NY - 1);
    }
  }
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
FS_PARALLEL_CONSTEXPR void apply_dirichlet_bconds(Field2D<Float, NX, NY, NGHOST>& field,
                                                  Float value) noexcept {
  static_assert(NGHOST > 0, "Expected at least one ghost cell.");

  FS_PARALLEL_FOR()
  for (Index j = -NGHOST; j < NY + NGHOST; ++j) {
    // LEFT
    for (Index i = -NGHOST; i < 0; ++i) {
      field(i, j) = value;
    }
    // RIGHT
    for (Index i = NX; i < NX + NGHOST; ++i) {
      field(i, j) = value;
    }
  }

  FS_PARALLEL_FOR()
  for (Index i = -NGHOST; i < NX + NGHOST; ++i) {
    // BOTTOM
    for (Index j = -NGHOST; j < 0; ++j) {
      field(i, j) = value;
    }
    // TOP
    for (Index j = NY; j < NY + NGHOST; ++j) {
      field(i, j) = value;
    }
  }
}

#endif  // FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
