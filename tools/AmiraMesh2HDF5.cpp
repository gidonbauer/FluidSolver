#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef FS_DISABLE_HDF
#error                                                                                             \
    "Cannot save data in HDF5 format because it was explicitly disabled via the FS_DISABLE_HDF macro."
#endif  // FS_DISABLE_HDF
#include <hdf5.h>
#include <hdf5_hl.h>

#include <Igor/Logging.hpp>
#include <Igor/MemoryToString.hpp>

#include "Container.hpp"

// = Trim from left ================================================================================
constexpr auto ltrim(std::string& s, const char* t = " \t\n\r\f\v") -> std::string& {
  s.erase(0, s.find_first_not_of(t));
  return s;
}

// = Trim from right ===============================================================================
constexpr auto rtrim(std::string& s, const char* t = " \t\n\r\f\v") -> std::string& {
  s.erase(s.find_last_not_of(t) + 1);
  return s;
}

// = Trim from left & right ========================================================================
constexpr auto trim(std::string& s, const char* t = " \t\n\r\f\v") -> std::string& {
  return ltrim(rtrim(s, t), t);
}

// = Remove prefix =================================================================================
constexpr auto remove_prefix(std::string& s, const std::string& prefix) -> std::string& {
  IGOR_ASSERT(s.starts_with(prefix), "Expected `{}` to start with `{}`.", s, prefix);
  s.erase(s.begin(), s.begin() + static_cast<long>(prefix.size()));
  return s;
}

// =================================================================================================
struct Dimension {
  Index nx, ny, nz;
};

struct BoundingBox {
  float x_min, x_max;
  float y_min, y_max;
  float z_min, z_max;
};

struct InterleavedView3D {
  static constexpr Index nd = 2;
  Dimension dim;
  float* data;

 private:
  [[nodiscard]] constexpr auto get_index(Index i, Index j, Index k, Index d) const -> Index {
    IGOR_ASSERT(0 <= i && i < dim.nx, "Index i={} is out of bounds with nx={}", i, dim.nx);
    IGOR_ASSERT(0 <= j && j < dim.ny, "Index j={} is out of bounds with ny={}", j, dim.ny);
    IGOR_ASSERT(0 <= k && k < dim.nz, "Index k={} is out of bounds with nz={}", k, dim.nz);
    IGOR_ASSERT(0 <= d && d < nd, "Index d={} is out of bounds with nd={}", d, nd);
    return d + nd * i + (nd * dim.nx) * j + (nd * dim.nx * dim.ny) * k;
  }

 public:
  [[nodiscard]] constexpr auto operator()(Index i, Index j, Index k, Index d) const -> float {
    return data[get_index(i, j, k, d)];
  }

  [[nodiscard]] constexpr auto operator()(Index i, Index j, Index k, Index d) -> float& {
    return data[get_index(i, j, k, d)];
  }
};

struct View2D {
  Index nx, ny;
  float* data;

 private:
  [[nodiscard]] constexpr auto get_index(Index i, Index j) const -> Index {
    IGOR_ASSERT(0 <= i && i < nx, "Index i={} is out of bounds with nx={}", i, nx);
    IGOR_ASSERT(0 <= j && j < ny, "Index j={} is out of bounds with ny={}", j, ny);
    return ny * i + j;
  }

 public:
  [[nodiscard]] constexpr auto operator()(Index i, Index j) const -> float {
    return data[get_index(i, j)];
  }

  [[nodiscard]] constexpr auto operator()(Index i, Index j) -> float& {
    return data[get_index(i, j)];
  }
};

// =================================================================================================
auto main(int argc, char** argv) -> int {
  using namespace std::string_literals;

  const char* mesh_filename = nullptr;
  std::string hdf5_filename{};
#define USAGE Igor::Error("Usage: {} [-o <output file>] <input file>", *argv)
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == "-o"s) {
      if (i + 1 >= argc) {
        USAGE;
        Igor::Error("       Did not provide output file for option `-o`.");
        return 1;
      }
      i             += 1;
      hdf5_filename  = argv[i];
    } else {
      mesh_filename = argv[i];
    }
  }
  if (mesh_filename == nullptr) {
    USAGE;
    Igor::Error("       Missing input file", argv[0]);
    return 1;
  }
  if (hdf5_filename.empty()) { hdf5_filename = mesh_filename + ".h5"s; }
#undef USAGE

  std::ifstream mesh_in(mesh_filename);
  if (!mesh_in) {
    Igor::Error("Could not open input file `{}`: {}", mesh_filename, std::strerror(errno));
    return 1;
  }

  static_assert(std::endian::native == std::endian::little,
                "Assume that the machine uses little endian.");
  Dimension dim{.nx = -1, .ny = -1, .nz = -1};
  BoundingBox bounding_box{
      .x_min = std::numeric_limits<float>::quiet_NaN(),
      .x_max = std::numeric_limits<float>::quiet_NaN(),
      .y_min = std::numeric_limits<float>::quiet_NaN(),
      .y_max = std::numeric_limits<float>::quiet_NaN(),
      .z_min = std::numeric_limits<float>::quiet_NaN(),
      .z_max = std::numeric_limits<float>::quiet_NaN(),
  };
  std::vector<float> data{};

  std::string line;
  std::stringstream strstream{};
  bool parse_parameter = false;
  for (int linenumber = 1; std::getline(mesh_in, line); ++linenumber) {
    line = trim(line);
    if (line.empty()) { continue; }

    if (parse_parameter) {
      if (line.starts_with("BoundingBox")) {
        line      = remove_prefix(line, "BoundingBox");
        strstream = std::stringstream(line);

        strstream >> bounding_box.x_min >> bounding_box.x_max >> bounding_box.y_min >>
            bounding_box.y_max >> bounding_box.z_min >> bounding_box.z_max;
        if (!strstream) {
          Igor::Error("Could not parse bounding box.");
          return 1;
        }
      } else if (line.starts_with("CoordType")) {
        if (!line.ends_with("\"uniform\"")) {
          Igor::Error("Only uniform coordinates are supported: got `{}`", line);
          return 1;
        }
      } else if (line == "}") {
        parse_parameter = false;
      } else {
        Igor::Todo("{}:{}: Unknown line `{}`", mesh_filename, linenumber, line);
      }

      continue;
    }

    if (line.starts_with("# AmiraMesh")) {
      line = remove_prefix(line, "# AmiraMesh");
      line = trim(line);
      if (!line.starts_with("BINARY-LITTLE-ENDIAN")) {
        Igor::Error("Expected binary little endian format: `{}`", line);
        return 1;
      }

      continue;
    } else if (line.starts_with("define Lattice")) {
      line      = remove_prefix(line, "define Lattice");
      strstream = std::stringstream(line);
      strstream >> dim.nx >> dim.ny >> dim.nz;
      if (!strstream) {
        Igor::Error("Could not parse lattice size.");
        return 1;
      }
    } else if (line.starts_with("Parameters")) {
      parse_parameter = true;
    } else if (line.starts_with("Lattice")) {
      if (line != "Lattice { float[2] Data } @1") {
        Igor::Error("Expected lattice layout with two interleaved float but got `{}`", line);
        return 1;
      }
    } else if (line == "# Data section follows") {
      if (!std::getline(mesh_in, line)) {
        Igor::Error("Could not read data section specifier.");
        return 1;
      }
      line = trim(line);
      if (line != "@1") {
        Igor::Error("Expected `@1` but got `{}`", line);
        return 1;
      }
      IGOR_ASSERT(dim.nx > 0 && dim.ny > 0 && dim.nz > 0,
                  "Invalid lattice size: {{ .nx = {}, .ny = {}, .nz = {} }}",
                  dim.nx,
                  dim.ny,
                  dim.nz);
      const size_t num_elem = static_cast<size_t>(dim.nx) * static_cast<size_t>(dim.ny) *
                              static_cast<size_t>(dim.nz) * 2;
      data.resize(num_elem);
      mesh_in.read(reinterpret_cast<char*>(data.data()),  // NOLINT
                   static_cast<std::streamsize>(num_elem * sizeof(float)));
    } else {
      Igor::Todo("{}:{}: Unknown line `{}`", mesh_filename, linenumber, line);
    }
  }

  InterleavedView3D view{.dim = dim, .data = data.data()};

  // = Save data as HDF5 ===========================================================================
  std::vector<float> data_2d(static_cast<size_t>(dim.nx * dim.ny));
  View2D view2d{.nx = dim.nx, .ny = dim.ny, .data = data_2d.data()};

  std::error_code ec;
  std::filesystem::remove(hdf5_filename, ec);
  if (ec) {
    Igor::Error("Could remove file `{}`: {}", hdf5_filename, ec.message());
    return 1;
  }
  const hid_t hdf5_file_id =
      H5Fcreate(hdf5_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write grid
  std::vector<float> grid{};
  auto save_single_grid = [&](size_t n, float min, float max, const char* dataset_name) {
    grid.resize(n);
    const float dx = (max - min) / static_cast<float>(n - 1);
    for (size_t i = 0; i < n; ++i) {
      grid[i] = min + static_cast<float>(i) * dx;
    }
    constexpr hsize_t RANK = 1;
    const auto DIM         = static_cast<hsize_t>(n);
    H5LTmake_dataset_float(hdf5_file_id, dataset_name, RANK, &DIM, grid.data());
  };

  save_single_grid(static_cast<size_t>(dim.nx), bounding_box.x_min, bounding_box.x_max, "/xcoords");
  save_single_grid(static_cast<size_t>(dim.ny), bounding_box.y_min, bounding_box.y_max, "/ycoords");
  save_single_grid(static_cast<size_t>(dim.nz), bounding_box.z_min, bounding_box.z_max, "/tcoords");

  // Write data
  for (Index k = 0; k < dim.nz; ++k) {
    const std::string group_name = "/"s + std::to_string(k);
    const hid_t group_id =
        H5Gcreate2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for (Index d = 0; d < InterleavedView3D::nd; ++d) {
      for (Index i = 0; i < dim.nx; ++i) {
        for (Index j = 0; j < dim.ny; ++j) {
          view2d(i, j) = view(i, j, k, d);
        }
      }

      constexpr hsize_t RANK              = 2;
      const std::array<hsize_t, RANK> DIM = {static_cast<hsize_t>(dim.nx),
                                             static_cast<hsize_t>(dim.ny)};
      const auto dataset_name = "/"s + std::to_string(k) + "/"s + (d == 0 ? "U"s : "V"s);
      H5LTmake_dataset_float(hdf5_file_id, dataset_name.c_str(), RANK, DIM.data(), view2d.data);
    }

    H5Gclose(group_id);
  }

  H5Fclose(hdf5_file_id);

  Igor::Info("Saved data in `{}`", hdf5_filename);
}
