#ifndef FLUID_SOLVER_XDMF_WRITER_HPP_
#define FLUID_SOLVER_XDMF_WRITER_HPP_

#ifndef FS_DISABLE_HDF

#include <fstream>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "Container.hpp"
#include "ForEach.hpp"

template <typename Float, Index NX, Index NY, Index NGHOST>
class XDMFWriter {
  std::string m_xdmf_path;
  std::string m_data_path;
  std::string m_data_filename;

  std::ofstream m_xdmf_out;
  // TODO: Maybe write each timestep in its own HDF5 file
  hid_t m_data_file_id;

  std::vector<std::string> m_scalar_names;
  std::vector<const Matrix<Float, NX, NY, NGHOST>*> m_scalar_values{};

  std::vector<std::string> m_vector_names;
  std::vector<std::array<const Matrix<Float, NX, NY, NGHOST>*, 2>> m_vector_values{};

  Matrix<Float, NX, NY, 0, Layout::F> m_local_storage{};

  // -----------------------------------------------------------------------------------------------
  void write_scalar(Index write_index,
                    const Matrix<Float, NX, NY, NGHOST>& scalar,
                    const std::string& name) {
    // - Write meta-data ---------------------------------------------------------------------------
    m_xdmf_out << Igor::detail::format(
                      R"(        <Attribute Name="{}" AttributeType="Scalar" Center="Cell">)", name)
               << '\n';
    m_xdmf_out
        << R"(          <DataItem Dimensions="&DimsZM; &DimsYM; &DimsXM;" NumberType="Float" Precision="8" Format="HDF">)"
        << '\n';
    m_xdmf_out << Igor::detail::format(
                      R"(            {}:/{}/{})", m_data_filename, write_index, name)
               << '\n';
    m_xdmf_out << R"(          </DataItem>)" << '\n';
    m_xdmf_out << R"(        </Attribute>)" << '\n';
    // - Write meta-data ---------------------------------------------------------------------------

    // Make sure that data is in Fortan order, this is incorrect for HDF5 but XDMF wants it...
    for_each_i(scalar, [&](Index i, Index j) { m_local_storage[i, j] = scalar[i, j]; });

    constexpr hsize_t RANK                  = 3;
    constexpr std::array<hsize_t, RANK> DIM = {NX, NY, 1};
    const auto dataset_name                 = Igor::detail::format("/{}/{}", write_index, name);
    H5LTmake_dataset_double(
        m_data_file_id, dataset_name.c_str(), RANK, DIM.data(), m_local_storage.get_data());
  }

 public:
  constexpr XDMFWriter(std::string xdmf_path,
                       std::string data_path,
                       const Vector<Float, NX + 1, NGHOST>* x,
                       const Vector<Float, NY + 1, NGHOST>* y)
      : m_xdmf_path(std::move(xdmf_path)),
        m_data_path(std::move(data_path)),
        m_data_filename(Igor::detail::strip_path(m_data_path)),
        m_xdmf_out(m_xdmf_path),
        m_data_file_id(H5Fcreate(m_data_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) {
    IGOR_ASSERT(x != nullptr, "x cannot be a nullptr.");
    IGOR_ASSERT(y != nullptr, "y cannot be a nullptr.");

    IGOR_ASSERT(m_xdmf_out.good(), "Could not open XDMF meta-data file.");
    // IGOR_ASSERT(m_data_file_id != 0, "Could not open HDF5 data file.");

    // - Begin XDMF file ---------------------------------------------------------------------------
    m_xdmf_out << R"(<?xml version="1.0" ?>)" << '\n';
    m_xdmf_out << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [)" << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsXM "{}">)", NX) << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsYM "{}">)", NY) << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsZM "{}">)", 1) << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsX  "{}">)", NX + 1) << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsY  "{}">)", NY + 1) << '\n';
    m_xdmf_out << Igor::detail::format(R"(  <!ENTITY DimsZ  "{}">)", 2) << '\n';
    m_xdmf_out << R"(]>)" << '\n';
    m_xdmf_out << R"(<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">)" << '\n';
    m_xdmf_out << R"(  <Domain>)" << '\n';
    m_xdmf_out
        << R"(    <Grid Name="TemporalCollection" GridType="Collection" CollectionType="Temporal">)"
        << '\n';
    m_xdmf_out << '\n';
    // - Begin XDMF file ---------------------------------------------------------------------------

    // Write grid
    auto dim = static_cast<hsize_t>(x->extent(0));
    H5LTmake_dataset_double(m_data_file_id, "/xcoords", 1, &dim, x->get_data());
    dim = static_cast<hsize_t>(y->extent(0));
    H5LTmake_dataset_double(m_data_file_id, "/ycoords", 1, &dim, y->get_data());
  }

  constexpr XDMFWriter(const XDMFWriter& other) noexcept                    = delete;
  constexpr XDMFWriter(XDMFWriter&& other) noexcept                         = delete;
  constexpr auto operator=(const XDMFWriter& other) noexcept -> XDMFWriter& = delete;
  constexpr auto operator=(XDMFWriter&& other) noexcept -> XDMFWriter&      = delete;
  constexpr ~XDMFWriter() noexcept {
    // - Finalize XDMF file ------------------------------------------------------------------------
    m_xdmf_out << R"(    </Grid>)" << '\n';
    m_xdmf_out << R"(  </Domain>)" << '\n';
    m_xdmf_out << R"(</Xdmf>)" << '\n';
    // - Finalize XDMF file ------------------------------------------------------------------------

    H5Fclose(m_data_file_id);
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void add_scalar(std::string name, const Matrix<Float, NX, NY, NGHOST>* value) {
    IGOR_ASSERT(value != nullptr, "value cannot be a nullptr.");
    m_scalar_names.emplace_back(std::move(name));
    m_scalar_values.push_back(value);
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void add_vector(std::string name,
                            const Matrix<Float, NX, NY, NGHOST>* x_value,
                            const Matrix<Float, NX, NY, NGHOST>* y_value) {
    IGOR_ASSERT(x_value != nullptr, "x_value cannot be a nullptr.");
    IGOR_ASSERT(y_value != nullptr, "y_value cannot be a nullptr.");
    m_vector_names.emplace_back(std::move(name));
    m_vector_values.push_back({x_value, y_value});
  }

  // -----------------------------------------------------------------------------------------------
  constexpr auto write(Float t) -> bool {
    static Index write_counter = 0;

    m_xdmf_out << Igor::detail::format(R"(      <Grid Name="State {}" GridType="Uniform">)",
                                       write_counter)
               << '\n';
    m_xdmf_out << Igor::detail::format(R"(        <Time Type="Single" Value="{:.6e}"/>)", t)
               << '\n';

    // - Write grid meta-data ----------------------------------------------------------------------
    m_xdmf_out
        << R"(        <Topology TopologyType="3DRectMesh" Dimensions="&DimsZ; &DimsY; &DimsX;" />)"
        << '\n';
    m_xdmf_out << R"(        <Geometry GeometryType="VXVYVZ">)" << '\n';
    m_xdmf_out
        << R"(          <DataItem Name="xcoords" Dimensions="&DimsX;" NumberType="Float" Precision="8" Format="HDF">)"
        << '\n';
    m_xdmf_out << Igor::detail::format(R"(            {}:/xcoords)", m_data_filename) << '\n';
    m_xdmf_out << R"(          </DataItem>)" << '\n';
    m_xdmf_out
        << R"(          <DataItem Name="ycoords" Dimensions="&DimsY;" NumberType="Float" Precision="8" Format="HDF">)"
        << '\n';
    m_xdmf_out << Igor::detail::format(R"(            {}:/ycoords)", m_data_filename) << '\n';
    m_xdmf_out << R"(          </DataItem>)" << '\n';
    m_xdmf_out
        << R"(          <DataItem Name="zcoords" Dimensions="&DimsZ;" NumberType="Float" Precision="8" Format="XML">)"
        << '\n';
    m_xdmf_out << R"(            -0.0001 0.0001)" << '\n';
    m_xdmf_out << R"(          </DataItem>)" << '\n';
    m_xdmf_out << R"(        </Geometry>)" << '\n';
    // - Write grid meta-data ----------------------------------------------------------------------

    // - Write data --------------------------------------------------------------------------------
    const auto group_name = Igor::detail::format("/{}", write_counter);
    const hid_t group_id =
        H5Gcreate2(m_data_file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write the time
    {
      constexpr hsize_t dim = 1;
      const auto name       = Igor::detail::format("/{}/time", write_counter);
      H5LTmake_dataset_double(m_data_file_id, name.c_str(), 1, &dim, &t);
    }

    for (size_t i = 0; i < m_scalar_names.size(); ++i) {
      write_scalar(write_counter, *m_scalar_values[i], m_scalar_names[i]);
    }

    for (size_t i = 0; i < m_vector_names.size(); ++i) {
      auto name = Igor::detail::format("{}_x", m_vector_names[i]);
      write_scalar(write_counter, *m_vector_values[i][0], name);
      name = Igor::detail::format("{}_y", m_vector_names[i]);
      write_scalar(write_counter, *m_vector_values[i][1], name);
    }

    H5Gclose(group_id);
    // - Write data --------------------------------------------------------------------------------

    // - Finalize meta-data ------------------------------------------------------------------------
    m_xdmf_out << R"(      </Grid>)" << '\n';
    // - Finalize meta-data ------------------------------------------------------------------------

    m_xdmf_out << '\n';
    write_counter += 1;
    return true;
  }
};

#else

#error HDF is disabled because the enviroment variable `HDF_DIR` is not set. If you want to use the XDMF writer, you need to define it.

// Dummy XDMFWriter to silence compiler error messages
template <typename Float, Index NX, Index NY>
class XDMFWriter {
 public:
  XDMFWriter(std::string,
             std::string,
             const Vector<Float, NX + 1, NGHOST>*,
             const Vector<Float, NY + 1, NGHOST>*);
  XDMFWriter(const XDMFWriter& other)                    = delete;
  XDMFWriter(XDMFWriter&& other)                         = delete;
  auto operator=(const XDMFWriter& other) -> XDMFWriter& = delete;
  auto operator=(XDMFWriter&& other) -> XDMFWriter&      = delete;
  ~XDMFWriter();

  void add_scalar(std::string, const Matrix<Float, NX, NY, NGHOST>*);
  void add_vector(std::string name,
                  const Matrix<Float, NX, NY, NGHOST>* x_value,
                  const Matrix<Float, NX, NY, NGHOST>* y_value);
  auto write(Float t) -> bool;
};

#endif  // FS_DISABLE_HDF

#endif  // FLUID_SOLVER_XDMF_WRITER_HPP_
