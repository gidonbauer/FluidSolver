#ifndef FLUID_SOLVER_IO_HPP_
#define FLUID_SOLVER_IO_HPP_

#include <bit>
#include <cstring>
#include <fstream>

#include <Igor/Logging.hpp>

#include "Container.hpp"

namespace detail {

// -------------------------------------------------------------------------------------------------
[[nodiscard]] constexpr auto interpret_as_bytes(double value)
    -> std::array<const char, sizeof(value)> {
  if constexpr (std::endian::native == std::endian::big) {
    return std::bit_cast<std::array<const char, sizeof(value)>>(value);
  }
  return std::bit_cast<std::array<const char, sizeof(value)>>(
      std::byteswap(std::bit_cast<uint64_t>(value)));
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
void write_scalar_vtk(std::ofstream& out,
                      const Matrix<Float, NX, NY>& scalar,
                      const std::string& name) {
  out << "SCALARS " << name << " double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (size_t j = 0; j < scalar.extent(1); ++j) {
    for (size_t i = 0; i < scalar.extent(0); ++i) {
      out.write(interpret_as_bytes(scalar[i, j]).data(), sizeof(scalar[i, j]));
    }
  }
  out << "\n\n";
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
void write_vector_vtk(std::ofstream& out,
                      const Matrix<Float, NX, NY>& x_comp,
                      const Matrix<Float, NX, NY>& y_comp,
                      const std::string& name) {
  out << "VECTORS " << name << " double\n";
  for (size_t j = 0; j < x_comp.extent(1); ++j) {
    for (size_t i = 0; i < x_comp.extent(0); ++i) {
      constexpr double z_comp_ij = 0.0;
      out.write(interpret_as_bytes(x_comp[i, j]).data(), sizeof(x_comp[i, j]));
      out.write(interpret_as_bytes(y_comp[i, j]).data(), sizeof(y_comp[i, j]));
      out.write(interpret_as_bytes(z_comp_ij).data(), sizeof(z_comp_ij));
    }
  }
  out << "\n\n";
}

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
[[nodiscard]] auto save_state_vtk(const std::string& output_dir,
                                  const Vector<Float, NX + 1>& x,
                                  const Vector<Float, NY + 1>& y,
                                  const Matrix<Float, NX, NY>& Ui,
                                  const Matrix<Float, NX, NY>& Vi,
                                  const Matrix<Float, NX, NY>& p,
                                  const Matrix<Float, NX, NY>& div,
                                  // const Matrix<Float, NX, NY>& vof,
                                  Float t) -> bool {
  static_assert(std::is_same_v<Float, double>, "Assumes Float=double");

  static size_t write_counter = 0;
  const auto filename = Igor::detail::format("{}/state_{:06d}.vtk", output_dir, write_counter++);
  std::ofstream out(filename);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  // = Write VTK header ============================================================================
  out << "# vtk DataFile Version 2.0\n";
  out << "State of FluidSolver at time t=" << t << '\n';
  out << "BINARY\n";

  // = Write grid ==================================================================================
  out << "DATASET STRUCTURED_GRID\n";
  out << "DIMENSIONS " << x.size() << ' ' << y.size() << " 1\n";
  out << "POINTS " << x.size() * y.size() << " double\n";
  for (size_t j = 0; j < y.size(); ++j) {
    for (size_t i = 0; i < x.size(); ++i) {
      constexpr double zk = 0.0;
      out.write(interpret_as_bytes(x[i]).data(), sizeof(x[i]));
      out.write(interpret_as_bytes(y[j]).data(), sizeof(y[j]));
      out.write(interpret_as_bytes(zk).data(), sizeof(zk));
    }
  }
  out << "\n\n";

  // = Write cell data =============================================================================
  IGOR_ASSERT(Ui.size() == Vi.size() && Ui.size() == p.size() && Ui.size() == div.size(),
              "Expected all fields to have the same size.");
  out << "CELL_DATA " << Ui.size() << '\n';
  write_scalar_vtk(out, p, "pressure");
  write_scalar_vtk(out, div, "divergence");
  write_vector_vtk(out, Ui, Vi, "velocity");
  // write_scalar_vtk(out, vof, "VOF");

  return out.good();
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
template <typename Float, size_t NX, size_t NY>
[[nodiscard]] auto save_state(const std::string& output_dir,
                              const Vector<Float, NX + 1>& x,
                              const Vector<Float, NY + 1>& y,
                              const Matrix<Float, NX, NY>& Ui,
                              const Matrix<Float, NX, NY>& Vi,
                              const Matrix<Float, NX, NY>& p,
                              const Matrix<Float, NX, NY>& div,
                              // const Matrix<Float, NX, NY>& vof,
                              Float t) -> bool {
  return detail::save_state_vtk(output_dir, x, y, Ui, Vi, p, div, /*vof,*/ t);
}

// -------------------------------------------------------------------------------------------------
template <typename Float>
[[nodiscard]] constexpr auto should_save(Float t, Float dt, Float dt_write, Float t_end) -> bool {
  constexpr Float DT_SAFE = 1e-6;
  return std::fmod(t + DT_SAFE * dt, dt_write) < dt * (1.0 - DT_SAFE) ||
         std::abs(t - t_end) < DT_SAFE;
}

#endif  // FLUID_SOLVER_IO_HPP_
