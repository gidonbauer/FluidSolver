#ifndef FLUID_SOLVER_IO_HPP_
#define FLUID_SOLVER_IO_HPP_

#include <cstring>
#include <filesystem>
#include <fstream>

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "FS.hpp"

#if defined(USE_VTK) || defined(FS_DISABLE_HDF)
#include "VTKWriter.hpp"
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = VTKWriter<Float, NX, NY, NGHOST>;
#else
#include "XDMFWriter.hpp"
template <typename Float, Index NX, Index NY, Index NGHOST>
using DataWriter = XDMFWriter<Float, NX, NY, NGHOST>;
#endif  // USE_VTK

namespace detail {

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
template <typename Float>
[[nodiscard]] constexpr auto should_save(Float t, Float dt, Float dt_write, Float t_end) -> bool {
  constexpr Float DT_SAFE      = 1e-6;
  static Float last_save_t     = -1.0;

  const bool dt_write_complete = std::fmod(t + DT_SAFE * dt, dt_write) < dt * (1.0 - DT_SAFE);
  const bool is_last           = std::abs(t - t_end) < DT_SAFE;
  const bool res               = dt_write_complete || is_last;
  if (res && is_last && std::abs(t - last_save_t) < DT_SAFE) { return false; }
  if (res) { last_save_t = t; }
  return res;
}

// -------------------------------------------------------------------------------------------------
#ifndef FS_BASE_DIR
#define FS_BASE_DIR "."
#endif  // FS_BASE_DIR
[[nodiscard]] auto
get_output_directory(std::string_view subdir   = "output",
                     std::string_view base_dir = FS_BASE_DIR,
                     std::source_location loc  = std::source_location::current()) noexcept
    -> std::string {
  // ===============
  auto strip_path = [](std::string_view full_path) -> std::string_view {
#if defined(WIN32) || defined(_WIN32)
#error "Not implemented yet: Requires different path separator ('\\') and potentially uses wchar"
#else
    constexpr char separator = '/';
#endif
    size_t counter = 0;
    for (char c : std::ranges::reverse_view(full_path)) {
      if (c == separator) { break; }
      ++counter;
    }

    return full_path.substr(full_path.size() - counter, counter);
  };

  // ===============
  auto strip_extension = [](std::string_view full_name) -> std::string_view {
    constexpr char separator = '.';
    size_t counter           = 0;
    for (char c : std::ranges::reverse_view(full_name)) {
      if (c == separator) { break; }
      ++counter;
    }

    return full_name.substr(0, full_name.size() - counter - 1);
  };

  // ===============
  try {
    return Igor::detail::format(
        "{}/{}/{}/", base_dir, subdir, strip_extension(strip_path(loc.file_name())));
  } catch (const std::exception& e) {
    Igor::Panic("Could not create output directory name.");
    std::unreachable();
  }
}

// -------------------------------------------------------------------------------------------------
[[nodiscard]] auto init_output_directory(std::string_view directory_name) noexcept -> bool {
  std::error_code ec;

  std::filesystem::remove_all(directory_name, ec);
  if (ec) {
    Igor::Warn("Could remove directory `{}`: {}", directory_name, ec.message());
    return false;
  }

  std::filesystem::create_directories(directory_name, ec);
  if (ec) {
    Igor::Warn("Could not create directory `{}`: {}", directory_name, ec.message());
    return false;
  }

  return true;
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index N, Index NGHOST>
[[nodiscard]] auto to_npy(const std::string& filename, const Vector<Float, N, NGHOST>& vector)
    -> bool {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  if (!detail::write_npy_header<Float, 1UZ>(
          out, {vector.extent(0) + 2 * NGHOST}, Layout::C, filename)) {
    return false;
  }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  if (!out.write(
          reinterpret_cast<const char*>(vector.get_data()),
          static_cast<std::streamsize>(static_cast<size_t>(vector.size()) * sizeof(Float)))) {
    Igor::Warn("Could not write data to `{}`: {}", filename, std::strerror(errno));
    return false;
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  return true;
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index M, Index N, Index NGHOST, Layout LAYOUT>
[[nodiscard]] auto to_npy(const std::string& filename,
                          const Matrix<Float, M, N, NGHOST, LAYOUT>& matrix) -> bool {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  if (!out) {
    Igor::Warn("Could not open file `{}`: {}", filename, std::strerror(errno));
    return false;
  }

  if (!detail::write_npy_header<Float, 2UZ>(
          out, {matrix.extent(0) + 2 * NGHOST, matrix.extent(1) + 2 * NGHOST}, LAYOUT, filename)) {
    return false;
  }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  if (!out.write(
          reinterpret_cast<const char*>(matrix.get_data()),
          static_cast<std::streamsize>(static_cast<size_t>(matrix.size()) * sizeof(Float)))) {
    Igor::Warn("Could not write data to `{}`: {}", filename, std::strerror(errno));
    return false;
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  return true;
}

// -------------------------------------------------------------------------------------------------
template <std::floating_point Float, Index M, Index N, Index NGHOST>
[[nodiscard]] auto to_npy(const std::string& directory, const FS<Float, M, N, NGHOST>& fs) -> bool {
  if (!init_output_directory(directory)) { return false; }

  if (!to_npy(Igor::detail::format("{}/x.npy", directory), fs.x)) { return false; }
  if (!to_npy(Igor::detail::format("{}/xm.npy", directory), fs.xm)) { return false; }
  if (!to_npy(Igor::detail::format("{}/y.npy", directory), fs.y)) { return false; }
  if (!to_npy(Igor::detail::format("{}/ym.npy", directory), fs.ym)) { return false; }

  if (!to_npy(Igor::detail::format("{}/visc.npy", directory), fs.visc)) { return false; }

  if (!to_npy(Igor::detail::format("{}/p.npy", directory), fs.p)) { return false; }
  if (!to_npy(Igor::detail::format("{}/p_jump_u_stag.npy", directory), fs.p_jump_u_stag)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/p_jump_v_stag.npy", directory), fs.p_jump_v_stag)) {
    return false;
  }

  if (!to_npy(Igor::detail::format("{}/rho_u_stag.npy", directory), fs.curr.rho_u_stag)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/rho_v_stag.npy", directory), fs.curr.rho_v_stag)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/U.npy", directory), fs.curr.U)) { return false; }
  if (!to_npy(Igor::detail::format("{}/V.npy", directory), fs.curr.V)) { return false; }

  if (!to_npy(Igor::detail::format("{}/rho_u_stag_old.npy", directory), fs.old.rho_u_stag)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/rho_v_stag_old.npy", directory), fs.old.rho_v_stag)) {
    return false;
  }
  if (!to_npy(Igor::detail::format("{}/U_old.npy", directory), fs.old.U)) { return false; }
  if (!to_npy(Igor::detail::format("{}/V_old.npy", directory), fs.old.V)) { return false; }

  return true;
}

#endif  // FLUID_SOLVER_IO_HPP_
