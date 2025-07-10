#ifndef FLUID_SOLVER_MONITOR_HPP_
#define FLUID_SOLVER_MONITOR_HPP_

#include <fstream>
#include <variant>

#include <Igor/Logging.hpp>

#include "Container.hpp"

template <typename Float>
class Monitor {
  static constexpr size_t MIN_LENGTH = 13;  // Min. value for format `.6e`

  std::ofstream m_out;
  std::string m_filename;

  bool m_wrote_header = false;

  std::vector<std::variant<Float const*, Index const*>> m_values{};
  std::vector<std::string> m_names;
  std::vector<size_t> m_lenghts;

  void write_header() {
    m_out << "| ";
    for (size_t i = 0; i < m_names.size(); ++i) {
      m_out << Igor::detail::format("{:^{}} | ", m_names[i], m_lenghts[i]);
    }
    m_out << '\n';

    m_out << '|';
    for (size_t length : m_lenghts) {
      m_out << Igor::detail::format("{:-<{}}|", "", length + 2);
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

  template <typename T>
  requires(std::is_same_v<T, Float> || std::is_same_v<T, Index>)
  constexpr void add_variable(T const* variable, std::string name) {
    IGOR_ASSERT(variable != nullptr, "Expected valid pointer but got nullptr.");
    m_values.push_back(variable);
    m_lenghts.push_back(std::max(name.size(), MIN_LENGTH));
    m_names.emplace_back(std::move(name));
  }

  void write() {
    if (m_names.empty()) { return; }
    if (!m_wrote_header) { write_header(); }

    m_out << "| ";
    for (size_t i = 0; i < m_values.size(); ++i) {
      const auto& val = m_values[i];
      size_t length   = m_lenghts[i];

      static_assert(std::variant_size_v<std::remove_cvref_t<decltype(m_values[0])>> == 2,
                    "Expected to hold exactly two alternatives.");
      switch (val.index()) {
        case 0:
          m_out << Igor::detail::format("{:^{}.6e} | ", *std::get<Float const*>(val), length);
          break;
        case 1:
          m_out << Igor::detail::format("{:^{}} | ", *std::get<Index const*>(val), length);
          break;
        default: Igor::Panic("Unreachable."); std::unreachable();
      }
    }
    m_out << '\n' << std::flush;
  }
};

#endif  // FLUID_SOLVER_MONITOR_HPP_
