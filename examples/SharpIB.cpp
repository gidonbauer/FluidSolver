#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Macros.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Geometry.hpp"
#include "IO.hpp"
#include "LinearSolver_StructHypre.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"

// = Config ========================================================================================
using Float              = double;

constexpr Float X_MIN    = 0.0;
constexpr Float X_MAX    = 5.0;
constexpr Float Y_MIN    = 0.0;
constexpr Float Y_MAX    = 1.0;

constexpr Index NY       = 128;
constexpr Index NX       = static_cast<Index>(NY * (X_MAX - X_MIN) / (Y_MAX - Y_MIN));
constexpr Index NGHOST   = 1;

constexpr Float T_END    = 5.0;
constexpr Float DT_MAX   = 1e-2;  // 1e-4;
constexpr Float CFL_MAX  = 0.5;
constexpr Float DT_WRITE = T_END / 100.0;

constexpr Float VISC     = 1e-3;
constexpr Float RHO      = 1.0;

#if 1
constexpr Circle wall        = {.x = 1.0, .y = 0.5, .r = 0.15};
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(wall.contains({.x = x, .y = y}));
};
constexpr auto normal_immersed_wall = [](Float x, Float y) -> std::array<Float, 2> {
  const auto d = std::sqrt(Igor::sqr(x - wall.x) + Igor::sqr(y - wall.y));
  return {
      (x - wall.x) / d,
      (y - wall.y) / d,
  };
};
#elif 0
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  constexpr Float R = 0.25;
  for (Float xi = 1.0; xi <= X_MAX + R; xi += 2.1 * R) {  // NOLINT
    Circle c{.x = xi, .y = Y_MIN, .r = R};
    if (is_in(c, x, y)) { return 1.0; }
    c.y = Y_MAX;
    if (is_in(c, x, y)) { return 1.0; }
  }
  return 0.0;
};
#elif 0
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  for (int i = 1; i < 5; ++i) {
    if (i - 0.05 <= x && x <= i + 0.05 && (0.6 <= y || y <= 0.4)) { return 1.0; }
  }
  return 0.0;
};
#else
constexpr auto immersed_wall = [](Float /*x*/, Float y) -> Float {
  return static_cast<Float>(y < 0.1 || y > 0.9);
};
#endif

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

constexpr auto U_in(Float y, [[maybe_unused]] Float t) -> Float {
  IGOR_ASSERT(t >= 0, "Expected t >= 0 but got t={:.6e}", t);
  constexpr Float height = Y_MAX - Y_MIN;
  constexpr Float U      = 1.5;
  return (4.0 * U * y * (height - y)) / Igor::sqr(height);

  // if (0.1 <= y && y <= 0.9) {
  //   constexpr Float height = (Y_MAX - 0.1) - (Y_MIN + 0.1);
  //   constexpr Float U      = 1.5;
  //   const Float y_         = y - 0.1;
  //   return (4.0 * U * y_ * (height - y_)) / Igor::sqr(height);
  // } else {
  //   return 0.0;
  // }
}

constexpr Float U_AVG =
    quadrature<64>([](Float y) { return U_in(y, 0.0); }, Y_MIN, Y_MAX) / (Y_MAX - Y_MIN);
constexpr Float Re = RHO * U_AVG * (Y_MAX - Y_MIN) / VISC;

// Channel flow
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet<Float>{.U = &U_in, .V = 0.0},
    .right  = Neumann{.clipped = true},
    .bottom = Dirichlet<Float>{.U = 0.0, .V = 0.0},
    .top    = Dirichlet<Float>{.U = 0.0, .V = 0.0},
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
                                  const Field2D<Float, NX + 1, NY, NGHOST>& ib_u_stag,
                                  const Field2D<Float, NX, NY + 1, NGHOST>& ib_v_stag,
                                  Float& mass,
                                  Float& momentum_x,
                                  Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for_each<0, NX, 0, NY>([&](Index i, Index j) {
    mass += ((1.0 - ib_u_stag(i, j)) * fs.curr.rho_u_stag(i, j) +
             (1.0 - ib_u_stag(i + 1, j)) * fs.curr.rho_u_stag(i + 1, j) +
             (1.0 - ib_v_stag(i, j)) * fs.curr.rho_v_stag(i, j) +
             (1.0 - ib_v_stag(i, j + 1)) * fs.curr.rho_v_stag(i, j + 1)) /
            4.0 * fs.dx * fs.dy;

    momentum_x +=
        ((1.0 - ib_u_stag(i, j)) * fs.curr.rho_u_stag(i, j) * fs.curr.U(i, j) +
         (1.0 - ib_u_stag(i + 1, j)) * fs.curr.rho_u_stag(i + 1, j) * fs.curr.U(i + 1, j)) /
        2.0 * fs.dx * fs.dy;
    momentum_y +=
        ((1.0 - ib_v_stag(i, j)) * fs.curr.rho_v_stag(i, j) * fs.curr.V(i, j) +
         (1.0 - ib_v_stag(i, j + 1)) * fs.curr.rho_v_stag(i, j + 1) * fs.curr.V(i, j + 1)) /
        2.0 * fs.dx * fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
enum IBBoundary : uint32_t {
  INSIDE  = 0b00000,
  LEFT    = 0b00001,
  RIGHT   = 0b00010,
  BOTTOM  = 0b00100,
  TOP     = 0b01000,
  OUTSIDE = 0b10000,
};

[[nodiscard]] auto is_boundary(const auto& xs, const auto& ys, Index i, Index j) -> uint32_t {
  if (immersed_wall(xs(i), ys(j)) < 1.0) { return IBBoundary::OUTSIDE; }

  uint32_t boundary = IBBoundary::INSIDE;
  if (immersed_wall(xs(i + 1), ys(j)) < 1.0) { boundary |= IBBoundary::RIGHT; }
  if (immersed_wall(xs(i - 1), ys(j)) < 1.0) { boundary |= IBBoundary::LEFT; }
  if (immersed_wall(xs(i), ys(j + 1)) < 1.0) { boundary |= IBBoundary::TOP; }
  if (immersed_wall(xs(i), ys(j - 1)) < 1.0) { boundary |= IBBoundary::BOTTOM; }
  return boundary;
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto get_extrapolated_velocity(
    const auto& xs, const auto& ys, Float dx, Float dy, const auto& vel, Index i, Index j)
    -> Float {
  auto get_weights = [](Float beta) -> std::array<Float, 3> {
#define LINEAR
#ifdef LINEAR
    return {
        1.0 / (1.0 - beta),
        -beta / (1.0 - beta),
        0.0,
    };
#else
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
#endif  // LINEAR
  };

  // const auto [normal_x, normal_y] = normal_immersed_wall(xs(i), ys(j));
  const auto normal    = normal_immersed_wall(xs(i), ys(j));
  const auto normal_x  = normal[0];
  const auto normal_y  = normal[1];
  const auto direction = [&]() {
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
        const Point pe{.x = xs(i), .y = ys(j)};
        const Point p1{.x = xs(i - 1), .y = ys(j)};
        const auto [intersect_x, intersect_y] = wall.intersect_line(pe, p1);
        beta                                  = std::abs(intersect_x - pe.x) / dx;
        U1                                    = vel(i - 1, j);
        U2                                    = vel(i - 2, j);
      }
      break;
    case IBBoundary::RIGHT:
      {
        const Point pe{.x = xs(i), .y = ys(j)};
        const Point p1{.x = xs(i + 1), .y = ys(j)};
        const auto [intersect_x, intersect_y] = wall.intersect_line(pe, p1);
        beta                                  = std::abs(intersect_x - pe.x) / dx;
        U1                                    = vel(i + 1, j);
        U2                                    = vel(i + 2, j);
      }
      break;
    case IBBoundary::BOTTOM:
      {
        const Point pe{.x = xs(i), .y = ys(j)};
        const Point p1{.x = xs(i), .y = ys(j - 1)};
        const auto [intersect_x, intersect_y] = wall.intersect_line(pe, p1);
        beta                                  = std::abs(intersect_y - pe.y) / dy;
        U1                                    = vel(i, j - 1);
        U2                                    = vel(i, j - 2);
      }
      break;
    case IBBoundary::TOP:
      {
        const Point pe{.x = xs(i), .y = ys(j)};
        const Point p1{.x = xs(i), .y = ys(j + 1)};
        const auto [intersect_x, intersect_y] = wall.intersect_line(pe, p1);
        beta                                  = std::abs(intersect_y - pe.y) / dy;
        U1                                    = vel(i, j + 1);
        U2                                    = vel(i, j + 2);
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
  Igor::Info("Re = {:.6e}", Re);

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  // = Allocate memory =============================================================================
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);
  LinearSolver_StructHypre<Float, NX, NY, NGHOST> ps(PRESSURE_TOL, PRESSURE_MAX_ITER);
  ps.set_pressure_operator(fs);

  Field2D<Float, NX, NY, NGHOST> Ui{};
  Field2D<Float, NX, NY, NGHOST> Vi{};
  Field2D<Float, NX, NY, NGHOST> div{};

  Field2D<Float, NX + 1, NY, NGHOST> drhoUdt{};
  Field2D<Float, NX, NY + 1, NGHOST> drhoVdt{};
  Field2D<Float, NX, NY, NGHOST> delta_p{};

  Field2D<Float, NX, NY, NGHOST> ib{};
  Field2D<Float, NX + 1, NY, NGHOST> ib_u_stag{};
  Field2D<Float, NX, NY + 1, NGHOST> ib_v_stag{};

  // IB forcing term
  Field2D<Float, NX, NY, NGHOST> fUi{};
  Field2D<Float, NX, NY, NGHOST> fVi{};
  Field2D<Float, NX + 1, NY, NGHOST> fU{};
  Field2D<Float, NX, NY + 1, NGHOST> fV{};

  // Observation variables
  Float t       = 0.0;
  Float dt      = DT_MAX;

  Float mass    = 0.0;
  Float mom_x   = 0.0;
  Float mom_y   = 0.0;

  Float U_max   = 0.0;
  Float V_max   = 0.0;

  Float div_max = 0.0;

  Float p_res   = 0.0;
  Index p_iter  = 0;
  Float p_max   = 0.0;
  // = Allocate memory =============================================================================

  // = Output ======================================================================================
  DataWriter<Float, NX, NY, NGHOST> data_writer(OUTPUT_DIR, &fs.x, &fs.y);
  data_writer.add_scalar("pressure", &fs.p);
  data_writer.add_scalar("divergence", &div);
  data_writer.add_vector("velocity", &Ui, &Vi);
  data_writer.add_scalar("Immersed-wall", &ib);
  data_writer.add_vector("IB-forcing", &fUi, &fVi);

  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&dt, "dt");

  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_max, "max(V)");

  monitor.add_variable(&div_max, "max(div)");

  monitor.add_variable(&p_max, "max(p)");
  monitor.add_variable(&p_res, "res(p)");
  monitor.add_variable(&p_iter, "iter(p)");

  monitor.add_variable(&mass, "mass");
  monitor.add_variable(&mom_x, "momentum (x)");
  monitor.add_variable(&mom_y, "momentum (y)");
  // = Output ======================================================================================

  // = Initialize immersed boundaries ==============================================================
  for_each_a<Exec::Parallel>(ib_u_stag, [&](Index i, Index j) {
    ib_u_stag(i, j) =
        quadrature(
            immersed_wall, fs.x(i) - fs.dx / 2.0, fs.x(i) + fs.dx / 2.0, fs.y(j), fs.y(j + 1)) /
        (fs.dx * fs.dy);
  });

  for_each_a<Exec::Parallel>(ib_v_stag, [&](Index i, Index j) {
    ib_v_stag(i, j) =
        quadrature(
            immersed_wall, fs.x(i), fs.x(i + 1), fs.y(j) - fs.dy / 2.0, fs.y(j) + fs.dy / 2.0) /
        (fs.dx * fs.dy);
  });
  for_each_a<Exec::Parallel>(ib, [&](Index i, Index j) {
    ib(i, j) =
        quadrature(immersed_wall, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
  });
  // interpolate_UV_staggered_field(ib_u_stag, ib_v_stag, ib);
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  for_each_a<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_a<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds, t);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
  p_max   = max(fs.p);
  calc_conserved_quantities_ib(fs, ib_u_stag, ib_v_stag, mass, mom_x, mom_y);
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
      apply_velocity_bconds(fs, bconds, t);
      // = Update flow field =======================================================================

      // = Correct the outflow =====================================================================
      Float inflow     = 0.0;
      Float outflow    = 0.0;
      Float mass_error = 0.0;
      calc_inflow_outflow(fs, inflow, outflow, mass_error);
      for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
        fs.curr.U(NX + NGHOST, j) -=
            mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 * NGHOST));
      });
      // = Correct the outflow =====================================================================

      // = IB forcing ==============================================================================
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        // Calculate
        const auto boundary = is_boundary(fs.x, fs.ym, i, j);
        if (boundary == IBBoundary::OUTSIDE) {
          fU(i, j) = 0.0;
        } else {
          Float u_target = 0.0;
          if (boundary != IBBoundary::INSIDE) {
            u_target = get_extrapolated_velocity(fs.x, fs.ym, fs.dx, fs.dy, fs.curr.U, i, j);
          }
          fU(i, j) = (u_target - fs.curr.U(i, j)) / dt;
        }

        // Apply
        fs.curr.U(i, j) += dt * fU(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        // Calculate
        const auto boundary = is_boundary(fs.xm, fs.y, i, j);
        if (boundary == IBBoundary::OUTSIDE) {
          fV(i, j) = 0.0;
        } else {
          Float v_target = 0.0;
          if (boundary != IBBoundary::INSIDE) {
            v_target = get_extrapolated_velocity(fs.xm, fs.y, fs.dx, fs.dy, fs.curr.V, i, j);
          }
          fV(i, j) = (v_target - fs.curr.V(i, j)) / dt;
        }

        // Apply
        fs.curr.V(i, j) += dt * fV(i, j);
      });
      interpolate_U(fU, fUi);
      interpolate_V(fV, fVi);
      // = IB forcing ==============================================================================

      // = Apply pressure correction ===============================================================
      calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
      Index local_p_iter = 0;
      ps.set_pressure_rhs(fs, div, dt);
      ps.solve(delta_p, &p_res, &local_p_iter);
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
    p_max   = max(fs.p);
    calc_conserved_quantities_ib(fs, ib_u_stag, ib_v_stag, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
