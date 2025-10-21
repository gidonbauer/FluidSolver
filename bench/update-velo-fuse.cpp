#include <chrono>

#include <Igor/Logging.hpp>
#include <Igor/MemoryToString.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Quadrature.hpp"

using Float                     = double;
constexpr Index NX              = 5 * (1 << 13);
constexpr Index NY              = 1 << 13;
constexpr Index NGHOST          = 1;

constexpr Float X_MIN           = 0.0;
constexpr Float X_MAX           = 5.0;
constexpr Float Y_MIN           = 0.0;
constexpr Float Y_MAX           = 1.0;

constexpr Float VISC_G          = 1e-6;
constexpr Float RHO_G           = 1.0;
constexpr Float VISC_L          = 1e-3;
constexpr Float RHO_L           = 1e3;
constexpr Float SURFACE_TENSION = 1.0 / 20.0;

constexpr Float DPDX            = 1e-2;

// -------------------------------------------------------------------------------------------------
constexpr auto vf0(Float x, Float y) -> Float {
  return static_cast<Float>(Igor::sqr(x - 2.5) + Igor::sqr(y - 0.5) < Igor::sqr(0.25));
}

// -------------------------------------------------------------------------------------------------
constexpr auto u_analytical = [](Float y, Float dy, Float dpdx) -> Float {
  // NOTE: Adjustment due to the ghost cells, the dirichlet boundary condition is now enforced
  //       in the ghost cell
  return dpdx / (2 * VISC_G) * (Igor::sqr(y) - y - (dy / 2.0 + Igor::sqr(dy / 2.0)));
};

// -------------------------------------------------------------------------------------------------
constexpr auto mean(const std::vector<Float>& values) -> Float {
  return std::reduce(values.cbegin(), values.cend(), Float{0}, std::plus<>{}) /
         static_cast<Float>(values.size());
}

// -------------------------------------------------------------------------------------------------
constexpr auto stddev(const std::vector<Float>& values, Float mean) -> Float {
  return std::sqrt(std::transform_reduce(values.cbegin(),
                                         values.cend(),
                                         Float{0},
                                         std::plus<>{},
                                         [&](Float value) { return Igor::sqr(value - mean); }) /
                   static_cast<Float>(values.size() - 1));
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY, Index NGHOST>
void update_velocity_fused(FS<Float, NX, NY, NGHOST>& fs, Float dt) {
  const auto rho_eps = calc_rho_eps(fs);

  // = On center mesh ========================
  const auto calc_FXU = [&](Index i, Index j) -> Float {
    // FXU = -rho*U*U + mu*(dUdx + dUdx - 2/3*(dUdx + dVdy)) - p
    //     = -rho*U^2 + mu*(2*dUdx -2/3*(dUdx + dVdy)) - p
    //     = -rho*U^2 + 2*mu*dUdx - p
    const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_u_stag(i, j),
                                                          fs.old.rho_u_stag(i + 1, j),
                                                          fs.curr.U(i, j),
                                                          fs.curr.U(i + 1, j),
                                                          fs.curr.U(i, j),
                                                          fs.curr.U(i + 1, j));
    const auto U_i                        = ((fs.curr.U(i + 1, j) + fs.curr.U(i, j)) / 2);
    const auto dUdx                       = (fs.curr.U(i + 1, j) - fs.curr.U(i, j)) / fs.dx;

    return -rho_i_hybrid * U_i_hybrid * U_i + 2.0 * fs.visc(i, j) * dUdx - fs.p(i, j);
  };

  // = On corner mesh ======================
  const auto calc_FYU = [&](Index i, Index j) -> Float {
    // FYU = -rho*U*V + mu*(dUdy + dVdx)
    const auto [rho_i_hybrid, U_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_u_stag(i, j - 1),
                                                          fs.old.rho_u_stag(i, j),
                                                          fs.curr.U(i, j - 1),
                                                          fs.curr.U(i, j),
                                                          fs.curr.V(i - 1, j),
                                                          fs.curr.V(i, j));
    const auto V_i                        = (fs.curr.V(i - 1, j) + fs.curr.V(i, j)) / 2;

    const auto visc_corner =
        (fs.visc(i, j) + fs.visc(i - 1, j) + fs.visc(i, j - 1) + fs.visc(i - 1, j - 1)) / 4.0;
    const auto dUdy = (fs.curr.U(i, j) - fs.curr.U(i, j - 1)) / fs.dy;
    const auto dVdx = (fs.curr.V(i, j) - fs.curr.V(i - 1, j)) / fs.dx;

    return -rho_i_hybrid * U_i_hybrid * V_i + visc_corner * (dUdy + dVdx);
  };

  // = On corner mesh ======================
  const auto calc_FXV = [&](Index i, Index j) -> Float {
    // FXV = -rho*U*V + mu*(dVdx + dUdy)
    const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_v_stag(i - 1, j),
                                                          fs.old.rho_v_stag(i, j),
                                                          fs.curr.V(i - 1, j),
                                                          fs.curr.V(i, j),
                                                          fs.curr.U(i, j - 1),
                                                          fs.curr.U(i, j));
    const auto U_i                        = (fs.curr.U(i, j) + fs.curr.U(i, j - 1)) / 2;

    const auto visc_corner =
        (fs.visc(i, j) + fs.visc(i - 1, j) + fs.visc(i, j - 1) + fs.visc(i - 1, j - 1)) / 4.0;
    const auto dUdy = (fs.curr.U(i, j) - fs.curr.U(i, j - 1)) / fs.dy;
    const auto dVdx = (fs.curr.V(i, j) - fs.curr.V(i - 1, j)) / fs.dx;

    return -rho_i_hybrid * U_i * V_i_hybrid + visc_corner * (dUdy + dVdx);
  };

  // = On center mesh ========================
  const auto calc_FYV = [&](Index i, Index j) {
    // FYV = -rho*V*V + mu*(dVdy + dVdy - 2/3*(dUdx + dVdy)) - p
    //     = -rho*V^2 + mu*(2*dVdy - 2/3*(dUdx + dVdy)) - p
    //     = -rho*V^2 + 2*mu*dVdy - p
    const auto [rho_i_hybrid, V_i_hybrid] = hybrid_interp(rho_eps,
                                                          fs.old.rho_v_stag(i, j),
                                                          fs.old.rho_v_stag(i, j + 1),
                                                          fs.curr.V(i, j),
                                                          fs.curr.V(i, j + 1),
                                                          fs.curr.V(i, j),
                                                          fs.curr.V(i, j + 1));
    const auto V_i                        = (fs.curr.V(i, j) + fs.curr.V(i, j + 1)) / 2;

    const auto dVdy                       = (fs.curr.V(i, j + 1) - fs.curr.V(i, j)) / fs.dy;

    return -rho_i_hybrid * V_i_hybrid * V_i + 2.0 * fs.visc(i, j) * dVdy - fs.p(i, j);
  };

  // =========================================
  const auto calc_drhoUdt = [&](Index i, Index j) {
    return (calc_FXU(i, j) - calc_FXU(i - 1, j)) / fs.dx +  //
           (calc_FYU(i, j + 1) - calc_FYU(i, j)) / fs.dy +  //
           fs.p_jump_u_stag(i, j);
  };

  // =========================================
  const auto calc_drhoVdt = [&](Index i, Index j) {
    return (calc_FXV(i + 1, j) - calc_FXV(i, j)) / fs.dx +  //
           (calc_FYV(i, j) - calc_FYV(i, j - 1)) / fs.dy +  //
           fs.p_jump_v_stag(i, j);
  };

  // = Calculate dmom[UV]dt ========================================================================
  for_each<0, NX + 1, 0, NY + 1, Exec::ParallelGPU>([&](Index i, Index j) {
    if (0 <= i && i < NX + 1 && 0 <= j && j < NY) {
      fs.curr.U(i, j) = (fs.old.rho_u_stag(i, j) * fs.old.U(i, j) + dt * calc_drhoUdt(i, j)) /
                        fs.curr.rho_u_stag(i, j);
    }

    if (0 <= i && i < NX && 0 <= j && j < NY + 1) {
      fs.curr.V(i, j) = (fs.old.rho_v_stag(i, j) * fs.old.V(i, j) + dt * calc_drhoVdt(i, j)) /
                        fs.curr.rho_v_stag(i, j);
    }
  });
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  Igor::Info("Approximate memory requirement for single field: {}",
             Igor::memory_to_string((NX + 2 * NGHOST) * (NY + 2 * NGHOST) * sizeof(Float)));

  FS<Float, NX, NY, NGHOST> fs{.visc_gas    = VISC_G,
                               .visc_liquid = VISC_L,
                               .rho_gas     = RHO_G,
                               .rho_liquid  = RHO_L,
                               .sigma       = SURFACE_TENSION};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  Matrix<Float, NX, NY, NGHOST> vf{};

  IGOR_TIME_SCOPE("Quadrature") {
    for_each_a<Exec::ParallelGPU>(vf, [&](Index i, Index j) {
      vf(i, j) = quadrature(vf0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
    });
  }
  calc_rho(vf, fs);
  calc_visc(vf, fs);

  for_each_a<Exec::Parallel>(
      fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = u_analytical(fs.ym(j), fs.dy, DPDX); });
  apply_neumann_bconds(fs.curr.U);
  for_each_i<Exec::Parallel>(fs.p,
                             [&](Index i, Index j) { fs.p(i, j) = DPDX * fs.dx + fs.p(i - 1, j); });
  apply_neumann_bconds(fs.p);

  const auto dt         = 1e-4;
  const size_t num_iter = 10;
  std::vector<double> runtimes(num_iter);
  IGOR_TIME_SCOPE("Measurements") {
    for (size_t i = 0; i < num_iter; ++i) {
      const auto t_begin = std::chrono::high_resolution_clock::now();
      update_velocity_fused(fs, dt);
      const auto t_duration = std::chrono::high_resolution_clock::now() - t_begin;
      runtimes[i]           = std::chrono::duration<double>(t_duration).count();
    }
  }

  const auto mean_runtime     = mean(runtimes);
  const auto stddev_runtime   = stddev(runtimes, mean_runtime);
  const auto [min_it, max_it] = std::minmax_element(runtimes.cbegin(), runtimes.cend());
  Igor::Info("NX = {}", NX);
  Igor::Info("NY = {}", NY);
  Igor::Info("Total cells = {}", NX * NY);
  Igor::detail::Time("mean_runtime   = {:.6f}s", mean_runtime);
  Igor::detail::Time("stddev_runtime = {:.6f}s", stddev_runtime);
  Igor::detail::Time("min_runtime    = {:.6f}s", *min_it);
  Igor::detail::Time("max_runtime    = {:.6f}s", *max_it);
}
