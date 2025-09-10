#include <chrono>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#include "FS.hpp"
#include "Quadrature.hpp"

using Float                     = double;
constexpr Index NX              = 5 * (1 << 12);
constexpr Index NY              = 1 << 12;
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
auto main() -> int {
  FS<Float, NX, NY, NGHOST> fs{.visc_gas    = VISC_G,
                               .visc_liquid = VISC_L,
                               .rho_gas     = RHO_G,
                               .rho_liquid  = RHO_L,
                               .sigma       = SURFACE_TENSION};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);

  Matrix<Float, NX, NY, NGHOST> vf{};
  Matrix<Float, NX + 1, NY, NGHOST> dmomUdt{};
  Matrix<Float, NX, NY + 1, NGHOST> dmomVdt{};

  IGOR_TIME_SCOPE("Quadrature") {
    for_each_a<Exec::ParallelGPU>(vf, [&](Index i, Index j) {
      vf(i, j) = quadrature(vf0, fs.x(i), fs.x(i + 1), fs.y(j), fs.y(j + 1)) / (fs.dx * fs.dy);
    });
  }
  calc_rho_and_visc(vf, fs);

  for_each_a<Exec::Parallel>(
      fs.curr.U, [&](Index i, Index j) { fs.curr.U(i, j) = u_analytical(fs.ym(j), fs.dy, DPDX); });
  apply_neumann_bconds(fs.curr.U);
  for_each_i<Exec::Parallel>(fs.p,
                             [&](Index i, Index j) { fs.p(i, j) = DPDX * fs.dx + fs.p(i - 1, j); });
  apply_neumann_bconds(fs.p);

  const size_t num_iter = 10;
  std::vector<double> runtimes(num_iter);
  IGOR_TIME_SCOPE("Measurements") {
    for (size_t i = 0; i < num_iter; ++i) {
      const auto t_begin = std::chrono::high_resolution_clock::now();
      calc_dmomdt(fs, dmomUdt, dmomVdt);
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
