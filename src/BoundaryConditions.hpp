#ifndef FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
#define FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_

#include "Container.hpp"
#include "ForEach.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
requires(NGHOST > 0)
struct FS;

// -------------------------------------------------------------------------------------------------
// TODO: Clipped Neumann?
enum class BCond : uint8_t { DIRICHLET, NEUMANN, PERIODIC, SYMMETRY };
enum : Index { LEFT, RIGHT, BOTTOM, TOP, NSIDES };

template <typename Float>
struct FlowBConds {
  std::array<BCond, NSIDES> types;
  std::array<Float, NSIDES> U;
  std::array<Float, NSIDES> V;
};

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void apply_velocity_bconds(FS<Float, NX, NY, NGHOST>& fs, const FlowBConds<Float>& bconds) {
  static_assert(NGHOST == 1, "Expected exactly one ghost cell.");

  // = Boundary conditions for U-component of velocity =============================================
  for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET: fs.curr.U(-NGHOST, j) = bconds.U[LEFT]; break;
      case BCond::NEUMANN:   fs.curr.U(-NGHOST, j) = fs.curr.U(0, j); break;
      case BCond::PERIODIC:  fs.curr.U(-NGHOST, j) = fs.curr.U(NX - 1, j); break;
      case BCond::SYMMETRY:
        fs.curr.U(-NGHOST, j) = -fs.curr.U(1, j);
        fs.curr.U(0, j)       = 0.0;
        break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET: fs.curr.U(NX + NGHOST, j) = bconds.U[RIGHT]; break;
      case BCond::NEUMANN:   fs.curr.U(NX + NGHOST, j) = fs.curr.U(NX, j); break;
      case BCond::PERIODIC:  fs.curr.U(NX + NGHOST, j) = fs.curr.U(1, j); break;
      case BCond::SYMMETRY:
        fs.curr.U(NX + NGHOST, j) = -fs.curr.U(NX - 1, j);
        fs.curr.U(NX, j)          = 0.0;
        break;
    }
  });

  for_each_a<Exec::Parallel>(fs.x, [&](Index i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET: fs.curr.U(i, -NGHOST) = bconds.U[BOTTOM]; break;
      case BCond::NEUMANN:   fs.curr.U(i, -NGHOST) = fs.curr.U(i, 0); break;
      case BCond::PERIODIC:  fs.curr.U(i, -NGHOST) = fs.curr.U(i, NY - 1); break;
      case BCond::SYMMETRY:  fs.curr.U(i, -NGHOST) = fs.curr.U(i, 0); break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET: fs.curr.U(i, NY + NGHOST - 1) = bconds.U[TOP]; break;
      case BCond::NEUMANN:   fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, NY - 1); break;
      case BCond::PERIODIC:  fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, 0); break;
      case BCond::SYMMETRY:  fs.curr.U(i, NY + NGHOST - 1) = fs.curr.U(i, NY - 1); break;
    }
  });

  // = Boundary conditions for V-component of velocity =============================================
  for_each_a<Exec::Parallel>(fs.y, [&](Index j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET: fs.curr.V(-NGHOST, j) = bconds.V[LEFT]; break;
      case BCond::NEUMANN:   fs.curr.V(-NGHOST, j) = fs.curr.V(0, j); break;
      case BCond::PERIODIC:  fs.curr.V(-NGHOST, j) = fs.curr.V(NX - 1, j); break;
      case BCond::SYMMETRY:  fs.curr.V(-NGHOST, j) = fs.curr.V(0, j); break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET: fs.curr.V(NX + NGHOST - 1, j) = bconds.V[RIGHT]; break;
      case BCond::NEUMANN:   fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(NX - 1, j); break;
      case BCond::PERIODIC:  fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(0, j); break;
      case BCond::SYMMETRY:  fs.curr.V(NX + NGHOST - 1, j) = fs.curr.V(NX - 1, j); break;
    }
  });

  for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET: fs.curr.V(i, -NGHOST) = bconds.V[BOTTOM]; break;
      case BCond::NEUMANN:   fs.curr.V(i, -NGHOST) = fs.curr.V(i, 0); break;
      case BCond::PERIODIC:  fs.curr.V(i, -NGHOST) = fs.curr.V(i, NY - 1); break;
      case BCond::SYMMETRY:
        fs.curr.V(i, -NGHOST) = -fs.curr.V(i, 1);
        fs.curr.V(i, 0)       = 0.0;
        break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET: fs.curr.V(i, NY + NGHOST) = bconds.V[TOP]; break;
      case BCond::NEUMANN:   fs.curr.V(i, NY + NGHOST) = fs.curr.V(i, NY); break;
      case BCond::PERIODIC:  fs.curr.V(i, NY + NGHOST) = fs.curr.V(i, 1); break;
      case BCond::SYMMETRY:
        fs.curr.V(i, NY + NGHOST) = -fs.curr.V(i, NY - 1);
        fs.curr.V(i, NY)          = 0.0;
        break;
    }
  });
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
