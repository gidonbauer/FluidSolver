#ifndef FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
#define FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_

#include "Container.hpp"
#include "ForEach.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
requires(NGHOST > 0)
struct FS;

// -------------------------------------------------------------------------------------------------
// TODO: Clipped Neumann?
enum class BCond : uint8_t { DIRICHLET, NEUMANN, PERIODIC };
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
      case BCond::DIRICHLET:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.U[i, j] = bconds.U[LEFT];
        }
        break;
      case BCond::NEUMANN:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.U[i, j] = fs.curr.U[0, j];
        }
        break;
      case BCond::PERIODIC:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.U[i, j] = fs.curr.U[NX + i, j];
        }
        break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET:
        for (Index i = NX + 1; i < NX + 1 + NGHOST; ++i) {
          fs.curr.U[i, j] = bconds.U[RIGHT];
        }
        break;
      case BCond::NEUMANN:
        for (Index i = NX + 1; i < NX + 1 + NGHOST; ++i) {
          fs.curr.U[i, j] = fs.curr.U[NX, j];
        }
        break;
      case BCond::PERIODIC:
        for (Index i = NX + 1; i < NX + 1 + NGHOST; ++i) {
          fs.curr.U[i, j] = fs.curr.U[0 + i - NX, j];
        }
        break;
    }
  });

  for_each_a<Exec::Parallel>(fs.x, [&](Index i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.U[i, j] = bconds.U[BOTTOM];
        }
        break;
      case BCond::NEUMANN:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.U[i, j] = fs.curr.U[i, 0];
        }
        break;
      case BCond::PERIODIC:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.U[i, j] = fs.curr.U[i, NY - 1];
        }
        break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET:
        for (Index j = NY; j < NY + NGHOST; ++j) {
          fs.curr.U[i, j] = bconds.U[TOP];
        }
        break;
      case BCond::NEUMANN:
        for (Index j = NY; j < NY + NGHOST; ++j) {
          fs.curr.U[i, j] = fs.curr.U[i, NY];
        }
        break;
      case BCond::PERIODIC:
        for (Index j = NY; j < NY + NGHOST; ++j) {
          fs.curr.U[i, j] = fs.curr.U[i, 0 + j - NY];
        }
        break;
    }
  });

  // = Boundary conditions for V-component of velocity =============================================
  for_each_a<Exec::Parallel>(fs.y, [&](Index j) {
    // LEFT
    switch (bconds.types[LEFT]) {
      case BCond::DIRICHLET:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.V[i, j] = bconds.V[LEFT];
        }
        break;
      case BCond::NEUMANN:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.V[i, j] = fs.curr.V[0, j];
        }
        break;
      case BCond::PERIODIC:
        for (Index i = -NGHOST; i < 0; ++i) {
          fs.curr.V[i, j] = fs.curr.V[NX + i, j];
        }
        break;
    }

    // RIGHT
    switch (bconds.types[RIGHT]) {
      case BCond::DIRICHLET:
        for (Index i = NX; i < NX + NGHOST; ++i) {
          fs.curr.V[i, j] = bconds.V[RIGHT];
        }
        break;
      case BCond::NEUMANN:
        for (Index i = NX; i < NX + NGHOST; ++i) {
          fs.curr.V[i, j] = fs.curr.V[NX - 1, j];
        }
        break;
      case BCond::PERIODIC:
        for (Index i = NX; i < NX + NGHOST; ++i) {
          fs.curr.V[i, j] = fs.curr.V[0 + i - NX, j];
        }
        break;
    }
  });

  for_each_a<Exec::Parallel>(fs.xm, [&](Index i) {
    // BOTTOM
    switch (bconds.types[BOTTOM]) {
      case BCond::DIRICHLET:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.V[i, j] = bconds.V[BOTTOM];
        }
        break;
      case BCond::NEUMANN:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.V[i, j] = fs.curr.V[i, 0];
        }
        break;
      case BCond::PERIODIC:
        for (Index j = -NGHOST; j < 0; ++j) {
          fs.curr.V[i, j] = fs.curr.V[i, NY + j];
        }
        break;
    }

    // TOP
    switch (bconds.types[TOP]) {
      case BCond::DIRICHLET:
        for (Index j = NY + 1; j < NY + 1 + NGHOST; ++j) {
          fs.curr.V[i, j] = bconds.V[TOP];
        }
        break;
      case BCond::NEUMANN:
        for (Index j = NY + 1; j < NY + 1 + NGHOST; ++j) {
          fs.curr.V[i, j] = fs.curr.V[i, NY];
        }
        break;
      case BCond::PERIODIC:
        for (Index j = NY + 1; j < NY + 1 + NGHOST; ++j) {
          fs.curr.V[i, j] = fs.curr.V[i, 0 + j - NY];
        }
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
      field[i, j] = field[0, j];
    }
    // RIGHT
    for (Index i = NX; i < NX + NGHOST; ++i) {
      field[i, j] = field[NX - 1, j];
    }
  });

  for_each<-NGHOST, NX + NGHOST, Exec::Parallel>([&](Index i) {
    // BOTTOM
    for (Index j = -NGHOST; j < 0; ++j) {
      field[i, j] = field[i, 0];
    }
    // TOP
    for (Index j = NY; j < NY + NGHOST; ++j) {
      field[i, j] = field[i, NY - 1];
    }
  });
}

#endif  // FLUID_SOLVER_BOUNDARY_CONDITIONS_HPP_
