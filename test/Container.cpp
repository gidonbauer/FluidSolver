#include "Container.hpp"

auto main() -> int {
  {
    Vector<double, 10> vec{};
    static_assert(vec.is_small, "Expect a Vector of size 10 to be allocated on the stack.");

    std::fill_n(vec.get_data(), vec.size(), 42.0);
    for (Index i = 0; i < vec.extent(0); ++i) {
      if (vec[i] != 42.0) { Igor::Warn("Expect value {} but got", 42.0, vec[i]); }
    }

    Matrix<double, 10, 10> mat{};
    static_assert(mat.is_small, "Expect a 10x10 Matrix to be allocated on the stack.");
    std::fill_n(mat.get_data(), mat.size(), 42.0);
    for (Index i = 0; i < mat.extent(0); ++i) {
      for (Index j = 0; j < mat.extent(1); ++j) {
        if (mat[i, j] != 42.0) { Igor::Warn("Expect value {} but got", 42.0, mat[i, j]); }
      }
    }
  }

  {
    Vector<double, 1000> vec{};
    static_assert(!vec.is_small, "Expect a Vector of size 1000 to not be allocated on the stack.");

    std::fill_n(vec.get_data(), vec.size(), 42.0);
    for (Index i = 0; i < vec.extent(0); ++i) {
      if (vec[i] != 42.0) { Igor::Warn("Expect value {} but got", 42.0, vec[i]); }
    }

    Matrix<double, 1000, 1000> mat{};
    static_assert(!mat.is_small, "Expect a 1000x1000 Matrix to not be allocated on the stack.");
    std::fill_n(mat.get_data(), mat.size(), 42.0);
    for (Index i = 0; i < mat.extent(0); ++i) {
      for (Index j = 0; j < mat.extent(1); ++j) {
        if (mat[i, j] != 42.0) { Igor::Warn("Expect value {} but got", 42.0, mat[i, j]); }
      }
    }
  }
}
