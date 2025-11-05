#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
// #define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// See: Vreman, A.W., 2020. Immersed boundary and overset grid methods assessed for Stokes flow due
// to an oscillating sphere. Journal of Computational Physics 423, 109783.

// = Config ========================================================================================
using Float              = double;

constexpr Index NX       = 5 * 128;
constexpr Index NY       = 128;
constexpr Index NGHOST   = 1;

constexpr Float X_MIN    = 0.0;
constexpr Float X_MAX    = 5.0;
constexpr Float Y_MIN    = 0.0;
constexpr Float Y_MAX    = 1.0;

constexpr Float T_END    = 4.0;
constexpr Float DT_MAX   = 1e-2;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = 1e-2;

constexpr Float U_BCOND  = 5.0;
constexpr Float VISC     = 1e-3;
constexpr Float RHO      = 1.0;

// #define IB_POLYGON
#ifndef IB_POLYGON
constexpr Float CX           = 1.0;
constexpr Float CY           = 0.5;
constexpr Float R0           = 0.1;
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
};
constexpr auto normal_immersed_wall = [](Float x, Float y) -> std::array<Float, 2> {
  const auto d = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));
  return {
      (x - CX) / d,
      (y - CY) / d,
  };
};
#else
struct Point {
  Float x, y;
};
[[nodiscard]] constexpr auto polygon_contains_point(const auto& polygon, const Point& p) noexcept
    -> bool {
  if (polygon.size() < 2) { return false; }

  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto& p1 = polygon[i];
    const auto& p2 = polygon[(i + 1) % polygon.size()];
    const auto n   = Point{
          .x = p1.y - p2.y,
          .y = -(p1.x - p2.x),
    };
    const auto v = n.x * (p.x - p1.x) + n.y * (p.y - p1.y);
    if (v < 0.0) { return false; }
  }
  return true;
}
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  constexpr Igor::StaticVector<Point, 8> polygon = {
      Point{.x = 0.9, .y = 0.5},
      Point{.x = 1.1, .y = 0.3},
      Point{.x = 2.0, .y = 0.6},
      Point{.x = 1.1, .y = 0.7},
  };
  return static_cast<Float>(polygon_contains_point(polygon, {.x = x, .y = y}));
};
#endif

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet{.U = U_BCOND, .V = 0.0},
    .right  = ClippedNeumann{},
    .bottom = Dirichlet{.U = 0.0, .V = 0.0},
    .top    = Dirichlet{.U = 0.0, .V = 0.0},
};
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
                         Float& inflow,
                         Float& outflow,
                         Float& mass_error) {
  inflow  = 0.0;
  outflow = 0.0;
  for_each_a(fs.ym, [&](Index j) {
    inflow  += fs.curr.rho_u_stag(-NGHOST, j) * fs.curr.U(-NGHOST, j);
    outflow += fs.curr.rho_u_stag(NX + NGHOST, j) * fs.curr.U(NX + NGHOST, j);
  });
  mass_error = outflow - inflow;
}

// -------------------------------------------------------------------------------------------------
void calc_conserved_quantities_ib(const FS<Float, NX, NY, NGHOST>& fs,
                                  Float& mass,
                                  Float& momentum_x,
                                  Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for_each<0, NX, 0, NY>([&](Index i, Index j) {
    mass += ((1.0 - immersed_wall(fs.x(i), fs.ym(j))) * fs.curr.rho_u_stag(i, j) +
             (1.0 - immersed_wall(fs.x(i + 1), fs.ym(j))) * fs.curr.rho_u_stag(i + 1, j) +
             (1.0 - immersed_wall(fs.xm(i), fs.y(j))) * fs.curr.rho_v_stag(i, j) +
             (1.0 - immersed_wall(fs.xm(i), fs.y(j + 1))) * fs.curr.rho_v_stag(i, j + 1)) /
            4.0 * fs.dx * fs.dy;

    momentum_x +=
        ((1.0 - immersed_wall(fs.x(i), fs.ym(j))) * fs.curr.rho_u_stag(i, j) * fs.curr.U(i, j) +
         (1.0 - immersed_wall(fs.x(i + 1), fs.ym(j))) * fs.curr.rho_u_stag(i + 1, j) *
             fs.curr.U(i + 1, j)) /
        2.0 * fs.dx * fs.dy;
    momentum_y +=
        ((1.0 - immersed_wall(fs.xm(i), fs.y(j))) * fs.curr.rho_v_stag(i, j) * fs.curr.V(i, j) +
         (1.0 - immersed_wall(fs.xm(i), fs.y(j + 1))) * fs.curr.rho_v_stag(i, j + 1) *
             fs.curr.V(i, j + 1)) /
        2.0 * fs.dx * fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
enum IBBoundary : uint32_t {
  INSIDE = 0b0000,
  LEFT   = 0b0001,
  RIGHT  = 0b0010,
  BOTTOM = 0b0100,
  TOP    = 0b1000,
};

[[nodiscard]] auto is_boundary(const auto& xs, const auto& ys, Index i, Index j) -> uint32_t {
  if (immersed_wall(xs(i), ys(j)) < 1.0) { return IBBoundary::INSIDE; }

  uint32_t boundary = IBBoundary::INSIDE;
  if (immersed_wall(xs(i + 1), ys(j)) < 1.0) { boundary |= IBBoundary::RIGHT; }
  if (immersed_wall(xs(i - 1), ys(j)) < 1.0) { boundary |= IBBoundary::LEFT; }
  if (immersed_wall(xs(i), ys(j + 1)) < 1.0) { boundary |= IBBoundary::TOP; }
  if (immersed_wall(xs(i), ys(j - 1)) < 1.0) { boundary |= IBBoundary::BOTTOM; }
  return boundary;
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto intersect_line_circle(std::array<Float, 2> p1,
                                         std::array<Float, 2> p2,
                                         std::array<Float, 2> c,
                                         Float r) {
  // See: https://mathworld.wolfram.com/Circle-LineIntersection.html
  enum : size_t { X, Y };

  p1[X]                  -= c[X];
  p1[Y]                  -= c[Y];
  p2[X]                  -= c[X];
  p2[Y]                  -= c[Y];

  const auto dx           = p2[X] - p1[X];
  const auto dy           = p2[Y] - p1[Y];
  const auto dr           = std::sqrt(dx * dx + dy * dy);
  const auto det          = p1[X] * p2[Y] - p2[X] * p1[Y];

  const auto inside_sqrt  = r * r * dr * dr - det * det;
  if (!(inside_sqrt >= 0.0)) {
    Igor::Panic("Line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}) and circle ({:.6e}, {:.6e}, R={:.6e}) "
                "do not intersect.",
                p1[X] + c[X],
                p1[Y] + c[Y],
                p2[X] + c[X],
                p2[Y] + c[Y],
                c[X],
                c[Y],
                r);
  }

  auto sign     = [](Float x) -> Float { return x < 0 ? -1.0 : 1.0; };
  std::array i1 = {
      (det * dy + sign(dy) * dx * std::sqrt(inside_sqrt)) / (dr * dr),
      (-det * dx + std::abs(dy) * std::sqrt(inside_sqrt)) / (dr * dr),
  };

  std::array i2 = {
      (det * dy - sign(dy) * dx * std::sqrt(inside_sqrt)) / (dr * dr),
      (-det * dx - std::abs(dy) * std::sqrt(inside_sqrt)) / (dr * dr),
  };

  auto on_finite_line = [&](const std::array<Float, 2>& i) -> bool {
    return std::min(p1[X], p2[X]) <= i[X] && i[X] <= std::max(p1[X], p2[X]) &&
           std::min(p1[Y], p2[Y]) <= i[Y] && i[Y] <= std::max(p1[Y], p2[Y]);
  };

  if (!(on_finite_line(i1) || on_finite_line(i2))) {
    Igor::Panic("None of the intersection points ({:.6e}, {:.6e}) and ({:.6e}, {:.6e}) is on the "
                "finite line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}).",
                i1[X] + c[X],
                i1[Y] + c[Y],
                i2[X] + c[X],
                i2[Y] + c[Y],
                p1[X] + c[X],
                p1[Y] + c[Y],
                p2[X] + c[X],
                p2[Y] + c[Y]);
  }
  if (on_finite_line(i1) && on_finite_line(i2)) {
    Igor::Panic("Both of the intersection points ({:.6e}, {:.6e}) and ({:.6e}, {:.6e}) are on the "
                "finite line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}).",
                i1[X] + c[X],
                i1[Y] + c[Y],
                i2[X] + c[X],
                i2[Y] + c[Y],
                p1[X] + c[X],
                p1[Y] + c[Y],
                p2[X] + c[X],
                p2[Y] + c[Y]);
  }

  if (on_finite_line(i1)) {
    i1[X] += c[X];
    i1[Y] += c[Y];
    return i1;
  } else {
    i2[X] += c[X];
    i2[Y] += c[Y];
    return i2;
  }
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto get_extrapolated_velocity(
    const auto& xs, const auto& ys, Float dx, Float dy, const auto& vel, Index i, Index j)
    -> Float {
  auto get_weights = [](Float beta) -> std::array<Float, 3> {
    constexpr Float BETA1 = 0.5;
    if (beta < BETA1) {
      return {
          2.0 / ((1.0 - beta) * (2.0 - beta)),
          -2.0 * beta / (1.0 - beta),
          beta / (2.0 - beta),
      };
    } else {
      const auto w0 = 2.0 / ((1.0 - BETA1) * (2.0 - BETA1));
      return {
          w0,
          2.0 - (2.0 - beta) * w0,
          -1.0 + (1.0 - beta) * w0,
      };
    }
  };

  const auto [normal_x, normal_y] = normal_immersed_wall(xs(i), ys(j));
  const auto direction            = [&]() {
    if (std::abs(normal_x) > std::abs(normal_y)) {
      return normal_x > 0.0 ? IBBoundary::RIGHT : IBBoundary::LEFT;
    } else {
      return normal_y > 0.0 ? IBBoundary::TOP : IBBoundary::BOTTOM;
    }
  }();
  // IGOR_ASSERT((direction & boundary) > 0, "Direction and boundary do not align.");

  const Float U0 = 0.0;  // Velocity at interface is set to zero
  Float U1       = 0.0;
  Float U2       = 0.0;
  Float beta     = 0.0;
  switch (direction) {
    case IBBoundary::LEFT:
      {
        const auto xe = xs(i);
        const auto ye = ys(j);
        const auto x1 = xs(i - 1);
        const auto y1 = ys(j);
        const auto [intersect_x, intersect_y] =
            intersect_line_circle({xe, ye}, {x1, y1}, {CX, CY}, R0);
        beta = std::abs(intersect_x - xe) / dx;
        U1   = vel(i - 1, j);
        U2   = vel(i - 2, j);
      }
      break;
    case IBBoundary::RIGHT:
      {
        const auto xe = xs(i);
        const auto ye = ys(j);
        const auto x1 = xs(i + 1);
        const auto y1 = ys(j);
        const auto [intersect_x, intersect_y] =
            intersect_line_circle({xe, ye}, {x1, y1}, {CX, CY}, R0);
        beta = std::abs(intersect_x - xe) / dx;
        U1   = vel(i + 1, j);
        U2   = vel(i + 2, j);
      }
      break;
    case IBBoundary::BOTTOM:
      {
        const auto xe = xs(i);
        const auto ye = ys(j);
        const auto x1 = xs(i);
        const auto y1 = ys(j - 1);
        const auto [intersect_x, intersect_y] =
            intersect_line_circle({xe, ye}, {x1, y1}, {CX, CY}, R0);
        beta = std::abs(intersect_y - ye) / dy;
        U1   = vel(i, j - 1);
        U2   = vel(i, j - 2);
      }
      break;
    case IBBoundary::TOP:
      {
        const auto xe = xs(i);
        const auto ye = ys(j);
        const auto x1 = xs(i);
        const auto y1 = ys(j + 1);
        const auto [intersect_x, intersect_y] =
            intersect_line_circle({xe, ye}, {x1, y1}, {CX, CY}, R0);
        beta = std::abs(intersect_y - ye) / dy;
        U1   = vel(i, j + 1);
        U2   = vel(i, j + 2);
      }
      break;
    case IBBoundary::INSIDE:
    default:                 Igor::Panic("Unreachable");
  }

  const auto [w0, w1, w2] = get_weights(beta);
  IGOR_ASSERT(std::abs(w0 + w1 + w2 - 1.0) < 1e-8,
              "Expected sum of weights to be 1.0 but is {:.6e}",
              w0 + w1 + w2);
  IGOR_ASSERT(std::abs(beta * w0 + w1 + 2.0 * w2) < 1e-8,
              "Expected beta*w0 + w1 + 2*w2 to be 0.0 but is {:.6e}",
              beta * w0 + w1 + 2.0 * w2);
  return w0 * U0 + w1 * U1 + w2 * U2;
};

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  PS ps(fs, PRESSURE_TOL, PRESSURE_MAX_ITER, PSSolver::PCG, PSPrecond::PFMG, PSDirichlet::NONE);

  Matrix<Float, NX, NY, NGHOST> Ui{};
  Matrix<Float, NX, NY, NGHOST> Vi{};
  Matrix<Float, NX, NY, NGHOST> div{};

  Matrix<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Matrix<Float, NX, NY, NGHOST> delta_p{};

  Matrix<Float, NX, NY, NGHOST> ib{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float mass    = 0.0;
  Float mom_x   = 0.0;
  Float mom_y   = 0.0;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;
  // Float div_L1        = 0.0;

  // Float p_max         = 0.0;
  Float p_res  = 0.0;
  Index p_iter = 0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("Immersed-wall", &ib);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");
  // monitor.add_variable(&div_L1, "L1(div)");

  // monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for_each_a(ib, [&](Index i, Index j) {
    ib(i, j) =
        quadrature(immersed_wall, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  for_each_a<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
    // fs.curr.U(i, j) = 1.0 - immersed_wall(fs.x(i), fs.ym(j));
    fs.curr.U(i, j) = 0.0;
  });
  for_each_a<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  calc_conserved_quantities_ib(fs, mass, mom_x, mom_y);
  if (!data_writer.write(t)) { return 1; }
  monitor.write();
  // = Initialize flow field =======================================================================

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    dt = adjust_dt(fs, CFL_MAX, DT_MAX);
    dt = std::min(dt, T_END - t);

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });

      // = IB forcing ==============================================================================
      for_each_i<Exec::Serial>(fs.curr.U, [&](Index i, Index j) {
        if (immersed_wall(fs.x(i), fs.ym(j)) > 0.0) { fs.curr.U(i, j) = 0.0; }
        const auto boundary = is_boundary(fs.x, fs.ym, i, j);
        if (boundary == IBBoundary::INSIDE) { return; }
        fs.curr.U(i, j) = get_extrapolated_velocity(fs.x, fs.ym, fs.dx, fs.dy, fs.curr.U, i, j);
      });
      for_each_i<Exec::Serial>(fs.curr.V, [&](Index i, Index j) {
        if (immersed_wall(fs.xm(i), fs.y(j)) > 0.0) { fs.curr.V(i, j) = 0.0; }
        const auto boundary = is_boundary(fs.xm, fs.y, i, j);
        if (boundary == IBBoundary::INSIDE) { return; }
        fs.curr.V(i, j) = get_extrapolated_velocity(fs.xm, fs.y, fs.dx, fs.dy, fs.curr.V, i, j);
      });
      // = IB forcing ==============================================================================

      apply_velocity_bconds(fs, bconds);

      // Correct the outflow
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });

      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);

      // = Apply pressure correction ===============================================================
      Index local_p_iter = 0;
      ps.setup(fs);
      ps.solve(fs, div, dt, delta_p, &p_res, &local_p_iter);
      p_iter += local_p_iter;

      shift_pressure_to_zero(fs.dx, fs.dy, delta_p);
      // Correct pressure
      for_each_a<Exec::Parallel>(fs.p, [&](Index i, Index j) { fs.p(i, j) += delta_p(i, j); });

      // Correct velocity
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        const auto dpdx  = (delta_p(i, j) - delta_p(i - 1, j)) / fs.dx;
        const auto rho   = fs.curr.rho_u_stag(i, j);
        fs.curr.U(i, j) -= dpdx * dt / rho;
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        const auto dpdy  = (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy;
        const auto rho   = fs.curr.rho_v_stag(i, j);
        fs.curr.V(i, j) -= dpdy * dt / rho;
      });
      // = Apply pressure correction ===============================================================
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
    U_max   = max(fs.curr.U);
    V_max   = max(fs.curr.V);
    div_max = max(div);
    calc_conserved_quantities_ib(fs, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
