#include "Container.hpp"

auto main() -> int {
  // ===============================================================================================
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

  // ===============================================================================================
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

  // ===============================================================================================
  {
    Vector<double, 10, 2> vec{};
    static_assert(vec.is_small,
                  "Expect a Vector of size 10 with 2 ghost cells to be allocated on the stack.");

    std::iota(vec.get_data(), vec.get_data() + vec.size(), -2.0);
    for (Index i = -2; i < vec.extent(0) + 2; ++i) {
      if (std::abs(vec[i] - static_cast<double>(i)) > 1e-12) {
        Igor::Warn("Expect value {} but got", 42.0, vec[i]);
      }
    }

    Matrix<double, 10, 5, 2> mat{};
    static_assert(mat.is_small,
                  "Expect a 10x5 Matrix with 2 ghost cells to be allocated on the stack.");
    std::iota(mat.get_data(), mat.get_data() + mat.size(), 0.0);
    Index counter = 0;
    for (Index i = -2; i < mat.extent(0) + 2; ++i) {
      for (Index j = -2; j < mat.extent(1) + 2; ++j) {
        if (std::abs(mat[i, j] - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat[i, j]);
        }
        counter += 1;
      }
    }

    Matrix<double, 10, 5, 2, Layout::F> mat_f{};
    static_assert(mat_f.is_small,
                  "Expect a 10x5 Matrix with 2 ghost cells to be allocated on the stack.");
    std::iota(mat_f.get_data(), mat_f.get_data() + mat_f.size(), 0.0);
    counter = 0;
    for (Index j = -2; j < mat_f.extent(1) + 2; ++j) {
      for (Index i = -2; i < mat_f.extent(0) + 2; ++i) {
        if (std::abs(mat_f[i, j] - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat_f[i, j]);
        }
        counter += 1;
      }
    }
  }

  // ===============================================================================================
  {
    Vector<double, 1000, 2> vec{};
    static_assert(
        !vec.is_small,
        "Expect a Vector of size 10 with 2 ghost cells to not be allocated on the stack.");

    std::iota(vec.get_data(), vec.get_data() + vec.size(), -2.0);
    for (Index i = -2; i < vec.extent(0) + 2; ++i) {
      if (std::abs(vec[i] - static_cast<double>(i)) > 1e-12) {
        Igor::Warn("Expect value {} but got {}", i, vec[i]);
      }
    }

    Matrix<double, 1000, 1000, 2> mat{};
    static_assert(!mat.is_small,
                  "Expect a 1000x1000 Matrix with 2 ghost cells to not be allocated on the stack.");
    std::iota(mat.get_data(), mat.get_data() + mat.size(), 0.0);
    Index counter = 0;
    for (Index i = -2; i < mat.extent(0) + 2; ++i) {
      for (Index j = -2; j < mat.extent(1) + 2; ++j) {
        if (std::abs(mat[i, j] - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat[i, j]);
        }
        counter += 1;
      }
    }

    Matrix<double, 1000, 1000, 2, Layout::F> mat_f{};
    static_assert(!mat_f.is_small,
                  "Expect a 1000x1000 Matrix with 2 ghost cells to not be allocated on the stack.");
    std::iota(mat_f.get_data(), mat_f.get_data() + mat_f.size(), 0.0);
    counter = 0;
    for (Index j = -2; j < mat_f.extent(1) + 2; ++j) {
      for (Index i = -2; i < mat_f.extent(0) + 2; ++i) {
        if (std::abs(mat_f[i, j] - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat_f[i, j]);
        }
        counter += 1;
      }
    }
  }
}
