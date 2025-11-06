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

// = Config ========================================================================================
using Float                     = double;

constexpr Index NX              = 128;
constexpr Index NY              = 128;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 1.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float T_END           = 1.0;
constexpr Float DT_MAX          = 1e-2;
constexpr Float CFL_MAX         = 0.5;
constexpr Float DT_WRITE        = 1e-2;

constexpr Float VISC            = 1e-3;
constexpr Float RHO             = 1.0;

constexpr int PRESSURE_MAX_ITER = 50;
constexpr Float PRESSURE_TOL    = 1e-6;

constexpr Index NUM_SUBITER     = 5;

#if 0
constexpr FlowBConds<Float> bconds{
    .left   = Dirichlet{.U = 1.0, .V = 0.0},
    .right  = Neumann{},
    .bottom = Dirichlet{.U = 0.0, .V = 0.0},
    .top    = Dirichlet{.U = 0.0, .V = 0.0},
};
#else
constexpr FlowBConds<Float> bconds{
    .left   = Neumann{.clipped = true},
    .right  = Neumann{.clipped = true},
    .bottom = Neumann{.clipped = true},
    .top    = Neumann{.clipped = true},
};
#endif
// = Config ========================================================================================

// -------------------------------------------------------------------------------------------------
// void calc_inflow_outflow(const FS<Float, NX, NY, NGHOST>& fs,
//                          Float& inflow,
//                          Float& outflow,
//                          Float& mass_error) {
//   inflow  = 0.0;
//   outflow = 0.0;
//   for_each_a(fs.ym, [&](Index j) {
//     inflow  += fs.curr.rho_u_stag(-NGHOST, j) * fs.curr.U(-NGHOST, j);
//     outflow += fs.curr.rho_u_stag(NX + NGHOST, j) * fs.curr.U(NX + NGHOST, j);
//   });
//   mass_error = outflow - inflow;
// }

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void calc_divergence_ib(const Matrix<Float, NX, NY, NGHOST>& ib,
                        const Matrix<Float, NX + 1, NY, NGHOST>& U,
                        const Matrix<Float, NX, NY + 1, NGHOST>& V,
                        Float dx,
                        Float dy,
                        Matrix<Float, NX, NY, NGHOST>& div) {
  for_each_a<Exec::Parallel>(div, [&](Index i, Index j) {
    div(i, j) = ((U(i + 1, j) - U(i, j)) / dx + (V(i, j + 1) - V(i, j)) / dy) * (1.0 - ib(i, j));
  });
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
void calc_ib(const FS<Float, NX, NY, NGHOST>& fs,
             auto&& immersed_wall,
             Float t,
             Matrix<Float, NX + 1, NY, NGHOST>& ib_u_stag,
             Matrix<Float, NX, NY + 1, NGHOST>& ib_v_stag,
             Matrix<Float, NX, NY, NGHOST>& ib) {
  for_each_a<Exec::Parallel>(ib_u_stag, [&](Index i, Index j) {
    ib_u_stag(i, j) = quadrature([&](Float x, Float y) { return immersed_wall(x, y, t); },
                                 fs.x(i) - fs.dx / 2.0,
                                 fs.x(i) + fs.dx / 2.0,
                                 fs.y(j),
                                 fs.y(j + 1)) /
                      (fs.dx * fs.dy);
  });
  for_each_a<Exec::Parallel>(ib_v_stag, [&](Index i, Index j) {
    ib_v_stag(i, j) = quadrature([&](Float x, Float y) { return immersed_wall(x, y, t); },
                                 fs.x(i),
                                 fs.x(i + 1),
                                 fs.y(j) - fs.dy / 2.0,
                                 fs.y(j) + fs.dy / 2.0) /
                      (fs.dx * fs.dy);
  });
  interpolate_UV_staggered_field(ib_u_stag, ib_v_stag, ib);
}

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
  Matrix<Float, NX + 1, NY, NGHOST> ib_u_stag{};
  Matrix<Float, NX, NY + 1, NGHOST> ib_v_stag{};
  Matrix<Float, NX + 1, NY, NGHOST> ib_u_forcing{};
  Matrix<Float, NX, NY + 1, NGHOST> ib_v_forcing{};
  Matrix<Float, NX, NY, NGHOST> fUi{};
  Matrix<Float, NX, NY, NGHOST> fVi{};

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
  data_writer.add_vector("IB_forcing", &fUi, &fVi);

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
  constexpr Float CX   = 0.5;
  constexpr Float CY   = 0.5;
  constexpr Float R0   = 0.1;
  constexpr Float DRDT = 0.1;
  auto r               = [](Float t) { return R0 + DRDT * t; };
  auto immersed_wall   = [&](Float x, Float y, Float t) -> Float {
    return static_cast<Float>(Igor::sqr(x - CX) + Igor::sqr(y - CY) <= Igor::sqr(r(t)));
  };

  calc_ib(fs, immersed_wall, t, ib_u_stag, ib_v_stag, ib);
  // = Initialize immersed boundaries ==============================================================

  // = Initialize flow field =======================================================================
  for_each_a<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = 0.0; });
  for_each_a<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) { fs.curr.V(i, j) = 0.0; });
  apply_velocity_bconds(fs, bconds);

  interpolate_U(fs.curr.U, Ui);
  interpolate_V(fs.curr.V, Vi);
  interpolate_U(ib_u_forcing, fUi);
  interpolate_V(ib_v_forcing, fVi);
  calc_divergence_ib(ib, fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
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

    // = Update IB field ===========================================================================
    calc_ib(fs, immersed_wall, t, ib_u_stag, ib_v_stag, ib);
    // = Update IB field ===========================================================================

    // Save previous state
    save_old_velocity(fs.curr, fs.old);

    p_iter = 0;
    for (Index sub_iter = 0; sub_iter < NUM_SUBITER; ++sub_iter) {
      calc_mid_time(fs.curr.U, fs.old.U);
      calc_mid_time(fs.curr.V, fs.old.V);

      // = Update flow field =======================================================================
      calc_dmomdt(fs, drhoUdt, drhoVdt);

      // = Calculate IB forcing term ===============================================================
      [[maybe_unused]] auto smoothing = []([[maybe_unused]] Float x) {
        return 1.0;
        // return std::exp(-Igor::sqr(x / 0.05));
      };

      for_each_i<Exec::Parallel>(ib_u_forcing, [&](Index i, Index j) {
        const auto x = fs.x(i);
        const auto y = fs.ym(j);
        const auto d = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));

        // const Float u_ib = d > 1e-8 ? DRDT *
        //                                   static_cast<Float>(ib_u_stag(i, j) > 1e-8 &&
        //                                                      ib_u_stag(i, j) < (1.0 - 1e-8)) *
        //                                   (x - CX) / d
        //                             : 0.0;

        const Float u_ib = d > 1e-8 ? DRDT * smoothing(d - r(t)) * (x - CX) / d : 0.0;

        // const Float u_ib = 2.0 * std::numbers::pi * DRDT * smoothing(d - r(t)) * (x - CX);

        // const Float u_ib = 2.0 * std::numbers::pi * DRDT * static_cast<Float>(d <= r(t)) * (x -
        // CX);

        ib_u_forcing(i, j) =
            (fs.old.rho_u_stag(i, j) * (u_ib - fs.old.U(i, j)) / dt - drhoUdt(i, j)) *
            ib_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(ib_v_forcing, [&](Index i, Index j) {
        const auto x = fs.xm(i);
        const auto y = fs.y(j);
        const auto d = std::sqrt(Igor::sqr(x - CX) + Igor::sqr(y - CY));

        // const Float v_ib = d > 1e-8 ? DRDT *
        //                                   static_cast<Float>(ib_v_stag(i, j) > 1e-2 &&
        //                                                      ib_v_stag(i, j) < (1.0 - 1e-2)) *
        //                                   (y - CY) / d
        //                             : 0.0;

        const Float v_ib = d > 1e-8 ? DRDT * smoothing(d - r(t)) * (y - CY) / d : 0.0;

        // const Float v_ib = 2.0 * std::numbers::pi * DRDT * smoothing(d - r(t)) * (y - CY);

        // const Float v_ib = 2.0 * std::numbers::pi * DRDT * static_cast<Float>(d <= r(t)) * (y -
        // CY);

        ib_v_forcing(i, j) =
            (fs.old.rho_v_stag(i, j) * (v_ib - fs.old.V(i, j)) / dt - drhoVdt(i, j)) *
            ib_v_stag(i, j);
      });
      // = Calculate IB forcing term ===============================================================

      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * drhoUdt(i, j)) /
                          fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * drhoVdt(i, j)) /
                          fs.curr.rho_v_stag(i, j);
      });

      // = Apply IB forcing term ===================================================================
      for_each_i<Exec::Parallel>(fs.curr.U, [&](Index i, Index j) {
        fs.curr.U(i, j) += ib_u_forcing(i, j) * dt / fs.curr.rho_u_stag(i, j);
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        fs.curr.V(i, j) += ib_v_forcing(i, j) * dt / fs.curr.rho_v_stag(i, j);
      });
      // = Apply IB forcing term ===================================================================

      apply_velocity_bconds(fs, bconds);

      // = Correct the outflow =====================================================================
      // Float inflow     = 0.0;
      // Float outflow    = 0.0;
      // Float mass_error = 0.0;
      // calc_inflow_outflow(fs, inflow, outflow, mass_error);
      // for_each_a<Exec::Parallel>(fs.ym, [&](Index j) {
      //   fs.curr.U(NX + NGHOST, j) -=
      //       mass_error / (fs.curr.rho_u_stag(NX + NGHOST, j) * static_cast<Float>(NY + 2 *
      //       NGHOST));
      // });
      // = Correct the outflow =====================================================================

      calc_divergence_ib(ib, fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
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
        fs.curr.U(i, j) -= (dpdx * dt / rho) * (1.0 - ib_u_stag(i, j));
      });
      for_each_i<Exec::Parallel>(fs.curr.V, [&](Index i, Index j) {
        const auto dpdy  = (delta_p(i, j) - delta_p(i, j - 1)) / fs.dy;
        const auto rho   = fs.curr.rho_v_stag(i, j);
        fs.curr.V(i, j) -= (dpdy * dt / rho) * (1.0 - ib_v_stag(i, j));
      });
      // = Apply pressure correction ===============================================================
    }

    t += dt;
    interpolate_U(fs.curr.U, Ui);
    interpolate_V(fs.curr.V, Vi);
    interpolate_U(ib_u_forcing, fUi);
    interpolate_V(ib_v_forcing, fVi);
    calc_divergence_ib(ib, fs.curr.U, fs.curr.V, fs.dx, fs.dy, div);
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
