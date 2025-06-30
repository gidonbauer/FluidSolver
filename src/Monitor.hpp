#ifndef FLUID_SOLVER_MONITOR_HPP_
#define FLUID_SOLVER_MONITOR_HPP_

#include <fstream>

#include <Igor/Logging.hpp>

template <typename Float>
class Monitor {
  std::ofstream m_out;
  std::string m_filename;

 public:
  Monitor(std::string filename)
      : m_out(filename),
        m_filename(std::move(filename)) {
    if (!m_out) {
      Igor::Panic("Could not open monitor file `{}`: {}", m_filename, std::strerror(errno));
    }
  }

  Monitor(const Monitor&)                    = delete;
  Monitor(Monitor&&)                         = delete;
  auto operator=(const Monitor&) -> Monitor& = delete;
  auto operator=(Monitor&&) -> Monitor&      = delete;
  ~Monitor() noexcept                        = default;

  // VOF
  Float max_volume_error;

  void write() {
    m_out << Igor::detail::format("max_volume_error = {:.6e}\n", max_volume_error);
    m_out << "------------------------------------------------------------\n";
  }
};

#endif  // FLUID_SOLVER_MONITOR_HPP_
