#include <cstddef>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

// #define FS_HYPRE_VERBOSE
#define FS_SILENCE_CONV_WARN

#include "FS.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Operators.hpp"
#include "PressureCorrection.hpp"
#include "Quadrature.hpp"
#include "Utility.hpp"
#if defined(USE_VTK) || defined(FS_DISABLE_HDF)
#include "VTKWriter.hpp"
template <typename Float, int NX, int NY, int NGHOST>
using DataWriter = VTKWriter<Float, NX, NY, NGHOST>;
#else
#include "XDMFWriter.hpp"
template <typename Float, int NX, int NY, int NGHOST>
using DataWriter = XDMFWriter<Float, NX, NY, NGHOST>;
#endif  // USE_VTK

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
constexpr Float U_0      = 0.0;
constexpr Float VISC     = 1e-3;
constexpr Float RHO      = 1.0;

#define IB_POLYGON
#ifndef IB_POLYGON
constexpr Float CX           = 1.0;
constexpr Float CY           = 0.5;
constexpr Float R0           = 0.1;
constexpr auto immersed_wall = [](Float x, Float y) -> Float {
  return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(R0));
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
    //        LEFT              RIGHT           BOTTOM            TOP
    .types = {BCond::DIRICHLET, BCond::NEUMANN, BCond::DIRICHLET, BCond::DIRICHLET},
    .U     = {U_BCOND, 0.0, 0.0, 0.0},
    .V     = {0.0, 0.0, 0.0, 0.0},
};

#ifndef FS_BASE_DIR
#define FS_BASE_DIR "."
#endif  // FS_BASE_DIR
constexpr auto OUTPUT_DIR = FS_BASE_DIR "/output/IB/";
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
                                  const Matrix<Float, NX + 1, NY, NGHOST>& ib_u_stag,
                                  const Matrix<Float, NX, NY + 1, NGHOST>& ib_v_stag,
                                  Float& mass,
                                  Float& momentum_x,
                                  Float& momentum_y) noexcept {
  mass       = 0.0;
  momentum_x = 0.0;
  momentum_y = 0.0;

  for_each<0, NX, 0, NY>([&](Index i, Index j) {
    mass += (ib_u_stag(i, j) * fs.curr.rho_u_stag(i, j) +
             ib_u_stag(i + 1, j) * fs.curr.rho_u_stag(i + 1, j) +
             ib_v_stag(i, j) * fs.curr.rho_v_stag(i, j) +
             ib_v_stag(i, j + 1) * fs.curr.rho_v_stag(i, j + 1)) /
            4.0 * fs.dx * fs.dy;

    momentum_x += (ib_u_stag(i, j) * fs.curr.rho_u_stag(i, j) * fs.curr.U(i, j) +
                   ib_u_stag(i + 1, j) * fs.curr.rho_u_stag(i + 1, j) * fs.curr.U(i + 1, j)) /
                  2.0 * fs.dx * fs.dy;
    momentum_y += (ib_v_stag(i, j) * fs.curr.rho_v_stag(i, j) * fs.curr.V(i, j) +
                   ib_v_stag(i, j + 1) * fs.curr.rho_v_stag(i, j + 1) * fs.curr.V(i, j + 1)) /
                  2.0 * fs.dx * fs.dy;
  });
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  // = Create output directory =====================================================================
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
  Matrix<Float, NX + 1, NY, NGHOST> ib_u_stag{};
  Matrix<Float, NX, NY + 1, NGHOST> ib_v_stag{};

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
  interpolate_UV_staggered_field(ib_u_stag, ib_v_stag, ib);
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  for_each_a<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = U_0; });
  for_each_a<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  calc_divergence(fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
  U_max   = max(fs.curr.U);
  V_max   = max(fs.curr.V);
  div_max = max(div);
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
        fs.curr.U(i, j) = (1.0 - ib_u_stag(i, j)) *
                          (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (1.0 - ib_v_stag(i, j)) *
                          (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });
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
    calc_conserved_quantities_ib(fs, ib_u_stag, ib_v_stag, mass, mom_x, mom_y);
    if (should_save(t, dt, DT_WRITE, T_END)) {
      if (!data_writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Solver finished successfully.");
}
