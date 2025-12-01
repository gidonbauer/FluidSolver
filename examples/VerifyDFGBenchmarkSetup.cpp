#include "DFGBenchmarkSetup.hpp"

auto main() -> int {
  FS<Float, NX, NY, NGHOST> fs{
      .visc_gas = VISC, .visc_liquid = VISC, .rho_gas = RHO, .rho_liquid = RHO};
  init_grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, fs);
  calc_rho(fs);
  calc_visc(fs);

  fill(fs.curr.U, 1000.0);
  fill(fs.curr.V, 1000.0);
  fill(fs.p, 1000.0);
  for_each_a(fs.p, [&](Index i, Index j) {
    if (immersed_wall(fs.xm(i), fs.ym(j)) > 0.0) { fs.p(i, j) = 0.0; }
  });

  Float t           = 0.0;
  const auto p_diff = calc_p_diff(fs);
  const auto C_L    = calc_C_L(fs, t);
  const auto C_D    = calc_C_D(fs, t);
  Igor::Info("p_diff = {:.6e}", p_diff);
  Igor::Info("C_L    = {:.6e}", C_L);
  Igor::Info("C_D    = {:.6e}", C_D);
}
