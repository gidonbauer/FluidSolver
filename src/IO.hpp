#ifndef FLUID_SOLVER_IO_HPP_
#define FLUID_SOLVER_IO_HPP_

#include <bit>
#include <fstream>

#include <Igor/Logging.hpp>
#include <Igor/MdArray.hpp>
#include <Igor/MdspanToNpy.hpp>

#include "Config.hpp"

namespace detail {

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto save_state_npy(const Igor::MdArray<Float, CENTERED_EXTENT>& Ui,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& Vi,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& p,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& div,
                                  Float t) {
  bool result = true;

  const auto Ui_filename = Igor::detail::format("{}/Ui_{:.6f}.npy", OUTPUT_DIR, t);
  result                 = Igor::mdspan_to_npy(Ui, Ui_filename) && result;

  const auto Vi_filename = Igor::detail::format("{}/Vi_{:.6f}.npy", OUTPUT_DIR, t);
  result                 = Igor::mdspan_to_npy(Vi, Vi_filename) && result;

  const auto p_filename = Igor::detail::format("{}/p_{:.6f}.npy", OUTPUT_DIR, t);
  result                = Igor::mdspan_to_npy(p, p_filename) && result;

  const auto div_filename = Igor::detail::format("{}/div_{:.6f}.npy", OUTPUT_DIR, t);
  result                  = Igor::mdspan_to_npy(div, div_filename) && result;

  return result;
}

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
void write_scalar_vtk(std::ofstream& out,
                      const Igor::MdArray<Float, CENTERED_EXTENT>& scalar,
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
void write_vector_vtk(std::ofstream& out,
                      const Igor::MdArray<Float, CENTERED_EXTENT>& x_comp,
                      const Igor::MdArray<Float, CENTERED_EXTENT>& y_comp,
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
[[nodiscard]] auto save_state_vtk(const Igor::MdArray<Float, NX_P1_EXTENT>& x,
                                  const Igor::MdArray<Float, NY_P1_EXTENT>& y,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& Ui,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& Vi,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& p,
                                  const Igor::MdArray<Float, CENTERED_EXTENT>& div,
                                  Float t) -> bool {
  static_assert(std::is_same_v<Float, double>, "Assumes Float=double");

  static size_t write_counter = 0;
  const auto filename = Igor::detail::format("{}/state_{}.vtk", OUTPUT_DIR, write_counter++);
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

  return out.good();
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto save_state([[maybe_unused]] const Igor::MdArray<Float, NX_P1_EXTENT>& x,
                              [[maybe_unused]] const Igor::MdArray<Float, NY_P1_EXTENT>& y,
                              const Igor::MdArray<Float, CENTERED_EXTENT>& Ui,
                              const Igor::MdArray<Float, CENTERED_EXTENT>& Vi,
                              const Igor::MdArray<Float, CENTERED_EXTENT>& p,
                              const Igor::MdArray<Float, CENTERED_EXTENT>& div,
                              Float t) -> bool {
#ifdef FS_SAVE_NUMPY
  return detail::save_state_npy(Ui, Vi, p, div, t);
#else
  return detail::save_state_vtk(x, y, Ui, Vi, p, div, t);
#endif
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] constexpr auto should_save(Float t, Float dt) -> bool {
  constexpr Float DT_SAFE = 1e-6;
  return std::fmod(t + DT_SAFE * dt, DT_WRITE) < dt * (1.0 - DT_SAFE) ||
         std::abs(t - T_END) < DT_SAFE;
}

#endif  // FLUID_SOLVER_IO_HPP_
