#ifndef FLUID_SOLVER_VTK_WRITER_HPP_
#define FLUID_SOLVER_VTK_WRITER_HPP_

#include <array>
#include <fstream>
#include <vector>

#include "Container.hpp"
#include "IO.hpp"

// TODO: Save as unstructured grid in VTKHDF file format
// (https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html)
template <typename Float, Index NX, Index NY>
class VTKWriter {
  std::string m_output_dir;

  const Vector<Float, NX + 1>* m_x;
  const Vector<Float, NY + 1>* m_y;

  std::vector<std::string> m_scalar_names;
  std::vector<const Matrix<Float, NX, NY>*> m_scalar_values{};

  std::vector<std::string> m_vector_names;
  std::vector<std::array<const Matrix<Float, NX, NY>*, 2>> m_vector_values{};

  // -----------------------------------------------------------------------------------------------
  void write_header(std::ofstream& out, Float t) {
    // = Write VTK header ====
    out << "# vtk DataFile Version 2.0\n";
    out << "State of FluidSolver at time t=" << t << '\n';
    out << "BINARY\n";

    // = Write grid ==========
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << m_x->size() << ' ' << m_y->size() << " 1\n";
    out << "POINTS " << m_x->size() * m_y->size() << " double\n";
    for (Index j = 0; j < m_y->size(); ++j) {
      for (Index i = 0; i < m_x->size(); ++i) {
        constexpr double zk = 0.0;
        out.write(detail::interpret_as_big_endian_bytes((*m_x)[i]).data(), sizeof((*m_x)[i]));
        out.write(detail::interpret_as_big_endian_bytes((*m_y)[j]).data(), sizeof((*m_y)[j]));
        out.write(detail::interpret_as_big_endian_bytes(zk).data(), sizeof(zk));
      }
    }
    out << "\n\n";

    // = Write cell data =====
    out << "CELL_DATA " << NX * NY << '\n';
  }

  // -----------------------------------------------------------------------------------------------
  void
  write_scalar(std::ofstream& out, const Matrix<Float, NX, NY>& scalar, const std::string& name) {
    out << "SCALARS " << name << " double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (Index j = 0; j < scalar.extent(1); ++j) {
      for (Index i = 0; i < scalar.extent(0); ++i) {
        out.write(detail::interpret_as_big_endian_bytes(scalar[i, j]).data(), sizeof(scalar[i, j]));
      }
    }
    out << "\n\n";
  }

  // -----------------------------------------------------------------------------------------------
  void write_vector(std::ofstream& out,
                    const Matrix<Float, NX, NY>& x_comp,
                    const Matrix<Float, NX, NY>& y_comp,
                    const std::string& name) {
    out << "VECTORS " << name << " double\n";
    for (Index j = 0; j < x_comp.extent(1); ++j) {
      for (Index i = 0; i < x_comp.extent(0); ++i) {
        constexpr double z_comp_ij = 0.0;
        out.write(detail::interpret_as_big_endian_bytes(x_comp[i, j]).data(), sizeof(x_comp[i, j]));
        out.write(detail::interpret_as_big_endian_bytes(y_comp[i, j]).data(), sizeof(y_comp[i, j]));
        out.write(detail::interpret_as_big_endian_bytes(z_comp_ij).data(), sizeof(z_comp_ij));
      }
    }
    out << "\n\n";
  }

 public:
  constexpr VTKWriter(std::string output_dir,
                      const Vector<Float, NX + 1>* x,
                      const Vector<Float, NY + 1>* y)
      : m_output_dir(std::move(output_dir)),
        m_x(x),
        m_y(y) {
    IGOR_ASSERT(m_x != nullptr, "x cannot be a nullptr.");
    IGOR_ASSERT(m_y != nullptr, "y cannot be a nullptr.");
  }

  constexpr VTKWriter(const VTKWriter& other) noexcept                    = delete;
  constexpr VTKWriter(VTKWriter&& other) noexcept                         = delete;
  constexpr auto operator=(const VTKWriter& other) noexcept -> VTKWriter& = delete;
  constexpr auto operator=(VTKWriter&& other) noexcept -> VTKWriter&      = delete;
  constexpr ~VTKWriter() noexcept                                         = default;

  constexpr void add_scalar(std::string name, const Matrix<Float, NX, NY>* value) {
    IGOR_ASSERT(value != nullptr, "value cannot be a nullptr.");
    m_scalar_names.emplace_back(std::move(name));
    m_scalar_values.push_back(value);
  }

  constexpr void add_vector(std::string name,
                            const Matrix<Float, NX, NY>* x_value,
                            const Matrix<Float, NX, NY>* y_value) {
    IGOR_ASSERT(x_value != nullptr, "x_value cannot be a nullptr.");
    IGOR_ASSERT(y_value != nullptr, "y_value cannot be a nullptr.");
    m_vector_names.emplace_back(std::move(name));
    m_vector_values.push_back({x_value, y_value});
  }

  constexpr auto write(Float t = -1.0) -> bool {
    static Index write_counter = 0;
    const auto filename =
        Igor::detail::format("{}/state_{:06d}.vtk", m_output_dir, write_counter++);
    std::ofstream out(filename);
    if (!out) {
      Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
      return false;
    }

    write_header(out, t);
    if (!out) {
      Igor::Warn("Could not write header to `{}`: {}", filename, std::strerror(errno));
      return false;
    }

    for (size_t i = 0; i < m_scalar_names.size(); ++i) {
      write_scalar(out, *m_scalar_values[i], m_scalar_names[i]);
    }

    for (size_t i = 0; i < m_vector_names.size(); ++i) {
      write_vector(out, *m_vector_values[i][0], *m_vector_values[i][1], m_vector_names[i]);
    }

    return out.good();
  }
};

#endif  // FLUID_SOLVER_VTK_WRITER_HPP_
