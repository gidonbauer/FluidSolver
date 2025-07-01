#ifndef FLUID_SOLVER_MONITOR_HPP_
#define FLUID_SOLVER_MONITOR_HPP_

#include <fstream>

#include <Igor/Logging.hpp>

template <typename Float>
class Monitor {
  std::ofstream m_out;
  std::string m_filename;

  bool m_wrote_header = false;
  size_t m_max_length = 13;  // Min. value for format `.6e`

  std::vector<Float const*> m_values{};
  std::vector<std::string> m_names;

  void write_header() {
    for (const auto& name : m_names) {
      m_max_length = std::max(name.length(), m_max_length);
    }

    m_out << "| ";
    for (const auto& name : m_names) {
      m_out << Igor::detail::format("{:^{}} | ", name, m_max_length);
    }
    m_out << '\n';

    m_out << '|';
    for (const auto& _ : m_names) {
      m_out << Igor::detail::format("{:-<{}}|", "", m_max_length + 2);
    }
    m_out << '\n';

    m_wrote_header = true;
  }

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

  constexpr void add_variable(Float const* variable, std::string name) {
    IGOR_ASSERT(variable != nullptr, "Expected valid pointer but got nullptr.");
    m_values.push_back(variable);
    m_names.emplace_back(std::move(name));
  }

  void write() {
    if (!m_wrote_header) { write_header(); }

    m_out << "| ";
    for (const auto& val : m_values) {
      m_out << Igor::detail::format("{:^{}.6e} | ", *val, m_max_length);
    }
    m_out << '\n' << std::flush;
  }
};

#endif  // FLUID_SOLVER_MONITOR_HPP_
