#ifndef FLUID_SOLVER_IO_HPP_
#define FLUID_SOLVER_IO_HPP_

#include <bit>
#include <cstring>
#include <fstream>

#include <Igor/Logging.hpp>

#include "Container.hpp"

namespace detail {

// -------------------------------------------------------------------------------------------------
[[nodiscard]] constexpr auto interpret_as_big_endian_bytes(double value)
    -> std::array<const char, sizeof(value)> {
  if constexpr (std::endian::native == std::endian::big) {
    return std::bit_cast<std::array<const char, sizeof(value)>>(value);
  }
  return std::bit_cast<std::array<const char, sizeof(value)>>(
      std::byteswap(std::bit_cast<uint64_t>(value)));
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void write_scalar_vtk(std::ofstream& out,
                      const Matrix<Float, NX, NY>& scalar,
                      const std::string& name) {
  out << "SCALARS " << name << " double 1\n";
  out << "LOOKUP_TABLE default\n";
  for (Index j = 0; j < scalar.extent(1); ++j) {
    for (Index i = 0; i < scalar.extent(0); ++i) {
      out.write(interpret_as_big_endian_bytes(scalar[i, j]).data(), sizeof(scalar[i, j]));
    }
  }
  out << "\n\n";
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
void write_vector_vtk(std::ofstream& out,
                      const Matrix<Float, NX, NY>& x_comp,
                      const Matrix<Float, NX, NY>& y_comp,
                      const std::string& name) {
  out << "VECTORS " << name << " double\n";
  for (Index j = 0; j < x_comp.extent(1); ++j) {
    for (Index i = 0; i < x_comp.extent(0); ++i) {
      constexpr double z_comp_ij = 0.0;
      out.write(interpret_as_big_endian_bytes(x_comp[i, j]).data(), sizeof(x_comp[i, j]));
      out.write(interpret_as_big_endian_bytes(y_comp[i, j]).data(), sizeof(y_comp[i, j]));
      out.write(interpret_as_big_endian_bytes(z_comp_ij).data(), sizeof(z_comp_ij));
    }
  }
  out << "\n\n";
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
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

  static Index write_counter = 0;
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
  for (Index j = 0; j < y.size(); ++j) {
    for (Index i = 0; i < x.size(); ++i) {
      constexpr double zk = 0.0;
      out.write(interpret_as_big_endian_bytes(x[i]).data(), sizeof(x[i]));
      out.write(interpret_as_big_endian_bytes(y[j]).data(), sizeof(y[j]));
      out.write(interpret_as_big_endian_bytes(zk).data(), sizeof(zk));
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

template <std::floating_point Float, Index RANK>
requires(RANK >= 1)
[[nodiscard]] auto write_npy_header(std::ostream& out,
                                    const std::array<Index, RANK>& extent,
                                    Layout layout,
                                    const std::string& filename) noexcept -> bool {
  using namespace std::string_literals;

  // Write magic string
  constexpr std::streamsize magic_string_len = 6;
  if (!out.write("\x93NUMPY", magic_string_len)) {
    Igor::Warn("Could not write magic string to  {}: {}", filename, std::strerror(errno));
    return false;
  }

  // Write format version, use 1.0
  constexpr std::streamsize version_len = 2;
  if (!out.write("\x01\x00", version_len)) {
    Igor::Warn("Could not write version number to  {}: {}", filename, std::strerror(errno));
    return false;
  }

  // Length of length entry
  constexpr std::streamsize header_len_len = 2;

  // Create header
  std::string header = "{"s;

  // Data type
  header += "'descr': '<f"s + std::to_string(sizeof(Float)) + "', "s;
  // Data order, Fortran order (column major) or C order (row major)
  header += "'fortran_order': "s + (layout == Layout::C ? "False"s : "True"s) + ", "s;
  // Data shape
  header += "'shape': ("s;
  for (size_t i = 0; i < static_cast<size_t>(RANK); ++i) {
    header += std::to_string(extent[i]) + ", "s;
  }
  header += "), "s;

  header += "}"s;

  // Pad header with spaces s.t. magic string, version, header length and header together are
  // divisible by 64
  for (auto preamble_len = magic_string_len + version_len + header_len_len + header.size() + 1;
       preamble_len % 64 != 0;
       preamble_len = magic_string_len + version_len + header_len_len + header.size() + 1) {
    header.push_back('\x20');
  }
  header.push_back('\n');

  // Write header length
  IGOR_ASSERT(
      header.size() <= std::numeric_limits<uint16_t>::max(),
      "Size cannot be larger than the max for an unsigned 16-bit integer as it is stored in one.");
  const auto header_len = static_cast<uint16_t>(header.size());
  if (!out.write(reinterpret_cast<const char*>(&header_len), header_len_len)) {  // NOLINT
    Igor::Warn("Could not write header length to  {}: {}", filename, std::strerror(errno));
    return false;
  }

  // Write header
  if (!out.write(header.data(), static_cast<std::streamsize>(header.size()))) {
    Igor::Warn("Could not write header to  {}: {}", filename, std::strerror(errno));
    return false;
  }

  return true;
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
template <typename Float, Index NX, Index NY>
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

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index N>
[[nodiscard]] auto to_npy(const std::string& filename, const Vector<Float, N>& vector) -> bool {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  if (!detail::write_npy_header<Float, 1UZ>(out, {vector.size()}, Layout::C, filename)) {
    return false;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  if (!out.write(reinterpret_cast<const char*>(vector.get_data()),
                 static_cast<std::streamsize>(vector.size() * sizeof(Float)))) {
    Igor::Warn("Could not write data to `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index M, Index N, Layout LAYOUT>
[[nodiscard]] auto to_npy(const std::string& filename, const Matrix<Float, M, N, LAYOUT>& matrix)
    -> bool {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  if (!detail::write_npy_header<Float, 2UZ>(
          out, {matrix.extent(0), matrix.extent(1)}, LAYOUT, filename)) {
    return false;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  if (!out.write(reinterpret_cast<const char*>(matrix.get_data()),
                 static_cast<std::streamsize>(matrix.size() * sizeof(Float)))) {
    Igor::Warn("Could not write data to `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  return true;
}

#endif  // FLUID_SOLVER_IO_HPP_
