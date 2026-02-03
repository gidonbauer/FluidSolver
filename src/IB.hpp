#ifndef FLUID_SOLVER_IB_HPP_
#define FLUID_SOLVER_IB_HPP_

#include <cstdint>

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "FS.hpp"
#include "ForEach.hpp"
#include "Geometry.hpp"

// See: Luchini, P., Gatti, D., Chiarini, A., Gattere, F., Atzori, M., Quadrio, M., 2025. A simple
//      and efficient second-order immersed-boundary method for the incompressible Navier–Stokes
//      equations. Journal of Computational Physics 539, 114245.
//      https://doi.org/10.1016/j.jcp.2025.114245

// -------------------------------------------------------------------------------------------------
enum Flow : uint8_t {
  FREE_FLOW      = 0b00000,
  WALL_TO_LEFT   = 0b00001,
  WALL_TO_RIGHT  = 0b00010,
  WALL_TO_BOTTOM = 0b00100,
  WALL_TO_TOP    = 0b01000,
  SOLID          = 0b10000,
};

template <typename FUNC, typename Float, Index NX, Index NY, Index NGHOST>
constexpr auto characterize_flow_regime(FUNC immersed_wall,
                                        const Field1D<Float, NX, NGHOST>& xs,
                                        const Field1D<Float, NY, NGHOST>& ys,
                                        Index i,
                                        Index j) -> uint8_t {
  if (immersed_wall(xs(i), ys(j)) > 0.0) { return Flow::SOLID; }

  uint8_t flow = Flow::FREE_FLOW;
  if (immersed_wall(xs(i + 1), ys(j)) > 0.0) { flow |= Flow::WALL_TO_RIGHT; }
  if (immersed_wall(xs(i - 1), ys(j)) > 0.0) { flow |= Flow::WALL_TO_LEFT; }
  if (immersed_wall(xs(i), ys(j + 1)) > 0.0) { flow |= Flow::WALL_TO_TOP; }
  if (immersed_wall(xs(i), ys(j - 1)) > 0.0) { flow |= Flow::WALL_TO_BOTTOM; }
  return flow;
}

// -------------------------------------------------------------------------------------------------
template <typename Shape, typename Float, Index NX, Index NY, Index NGHOST>
constexpr void calc_ib_correction_shape(const Shape& wall,
                                        Float dx,
                                        Float dy,
                                        const Field1D<Float, NX, NGHOST>& xs,
                                        const Field1D<Float, NY, NGHOST>& ys,
                                        Field2D<Float, NX, NY, NGHOST>& ib_corr) {
  auto immersed_wall = [&](Float x, Float y) { return wall.contains({.x = x, .y = y}); };
  for_each_i<Exec::Parallel>(ib_corr, [&](Index i, Index j) {
    const auto flow = characterize_flow_regime(immersed_wall, xs, ys, i, j);
    if (flow == Flow::FREE_FLOW) { return; }
    if (flow == Flow::SOLID) {
      ib_corr(i, j) = std::numeric_limits<Float>::infinity();
      return;
    }

    [[maybe_unused]] constexpr Float TOL = 1e-6;
    const Point p_center                 = {.x = xs(i), .y = ys(j)};
    if ((flow & Flow::WALL_TO_RIGHT) > 0) {
      const Point p_other   = {.x = xs(i + 1), .y = ys(j)};
      const Point intersect = wall.intersect_line(p_center, p_other);
      const Float dist      = intersect.x - p_center.x;
      IGOR_ASSERT(0.0 < dist && dist < dx + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dx,
                  dist);
      const Float lambda  = (dx - dist) / (dist * dx * dx);
      ib_corr(i, j)      += lambda;
    }
    if ((flow & Flow::WALL_TO_LEFT) > 0) {
      const Point p_other   = {.x = xs(i - 1), .y = ys(j)};
      const Point intersect = wall.intersect_line(p_center, p_other);
      const Float dist      = p_center.x - intersect.x;
      IGOR_ASSERT(0.0 < dist && dist < dx + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dx,
                  dist);
      const Float lambda  = (dx - dist) / (dist * dx * dx);
      ib_corr(i, j)      += lambda;
    }
    if ((flow & Flow::WALL_TO_TOP) > 0) {
      const Point p_other   = {.x = xs(i), .y = ys(j + 1)};
      const Point intersect = wall.intersect_line(p_center, p_other);
      const Float dist      = intersect.y - p_center.y;
      IGOR_ASSERT(0.0 < dist && dist < dy + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dy,
                  dist);
      const Float lambda  = (dy - dist) / (dist * dy * dy);
      ib_corr(i, j)      += lambda;
    }
    if ((flow & Flow::WALL_TO_BOTTOM) > 0) {
      const Point p_other   = {.x = xs(i), .y = ys(j - 1)};
      const Point intersect = wall.intersect_line(p_center, p_other);
      const Float dist      = p_center.y - intersect.y;
      IGOR_ASSERT(0.0 < dist && dist < dy + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dy,
                  dist);
      const Float lambda  = (dy - dist) / (dist * dy * dy);
      ib_corr(i, j)      += lambda;
    }
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
constexpr void
correct_velocity_ib_implicit_euler(const Field2D<Float, NX + 1, NY, NGHOST>& ib_corr_u_stag,
                                   const Field2D<Float, NX, NY + 1, NGHOST>& ib_corr_v_stag,
                                   Float dt,
                                   FS<Float, NX, NY, NGHOST>& fs) {
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
    const Float visc  = (fs.visc(i, j) + fs.visc(i - 1, j)) / 2.0;
    const Float rho   = fs.curr.rho_u_stag(i, j);
    fs.curr.U(i, j)  /= 1.0 + dt * visc / rho * ib_corr_u_stag(i, j);
  });
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
    const Float visc  = (fs.visc(i, j) + fs.visc(i, j - 1)) / 2.0;
    const Float rho   = fs.curr.rho_v_stag(i, j);
    fs.curr.V(i, j)  /= 1.0 + dt * visc / rho * ib_corr_v_stag(i, j);
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void update_velocity_ib_semi_analytical(const Field2D<Float, NX + 1, NY, NGHOST>& drhoUdt,
                                        const Field2D<Float, NX, NY + 1, NGHOST>& drhoVdt,
                                        Float dt,
                                        const Field2D<Float, NX + 1, NY, NGHOST>& ib_corr_u_stag,
                                        const Field2D<Float, NX, NY + 1, NGHOST>& ib_corr_v_stag,
                                        FS<Float, NX, NY, NGHOST>& fs) {
  // From Luchini et al.
  // A*U^(n+1) - B*U^n = C*F^n
  // U^(n+1) = (C*F^n + B*U^n) / A
  //
  // B = (lambda*dt) / (exp(lambda*dt) - 1)
  // A = lambda*dt + B
  // C = dt

  auto calc_coefficients = [](Float lambda, Float dt, Float& A, Float& B, Float& C) {
    B = std::abs(lambda) < 1e-6 ? 1.0 : (lambda * dt) / (std::exp(lambda * dt) - 1.0);
    IGOR_ASSERT(
        0 <= B && B <= 1, "Expected B in [0, 1] but got B = {:.6e} (lambda = {:.6e})", B, lambda);
    A = lambda * dt + B;
    C = dt;
  };

  // = U =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
    const Float visc   = (fs.visc(i, j) + fs.visc(i - 1, j)) / 2.0;
    const Float rho    = fs.curr.rho_u_stag(i, j);
    const Float lambda = visc / rho * ib_corr_u_stag(i, j);
    if (std::isinf(lambda)) {
      fs.curr.U(i, j) = 0.0;
      return;
    }

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);

    fs.curr.U(i, j) = (B * fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + C * drhoUdt(i, j)) /
                      (A * fs.curr.rho_u_stag(i, j));
  });

  // = V =======================================================================
  for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
    const Float visc   = (fs.visc(i, j) + fs.visc(i, j - 1)) / 2.0;
    const Float rho    = fs.curr.rho_v_stag(i, j);
    const Float lambda = visc / rho * ib_corr_v_stag(i, j);
    if (std::isinf(lambda)) {
      fs.curr.V(i, j) = 0.0;
      return;
    }

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);

    fs.curr.V(i, j) = (B * fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + C * drhoVdt(i, j)) /
                      (A * fs.curr.rho_v_stag(i, j));
  });
}

#endif  // FLUID_SOLVER_IB_HPP_
