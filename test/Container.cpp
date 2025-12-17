#include "Container.hpp"

auto main() -> int {
  // ===============================================================================================
  {
    Field1D<double, 5> vec{};

    std::fill_n(vec.get_data(), vec.size(), 42.0);
    for (Index i = 0; i < vec.extent(0); ++i) {
      if (vec(i) != 42.0) { Igor::Warn("Expect value {} but got", 42.0, vec(i)); }
    }

    Field2D<double, 5, 5> mat{};
    std::fill_n(mat.get_data(), mat.size(), 42.0);
    for (Index i = 0; i < mat.extent(0); ++i) {
      for (Index j = 0; j < mat.extent(1); ++j) {
        if (mat(i, j) != 42.0) { Igor::Warn("Expect value {} but got", 42.0, mat(i, j)); }
      }
    }
  }

  // ===============================================================================================
  {
    Field1D<double, 1000> vec{};

    std::fill_n(vec.get_data(), vec.size(), 42.0);
    for (Index i = 0; i < vec.extent(0); ++i) {
      if (vec(i) != 42.0) { Igor::Warn("Expect value {} but got", 42.0, vec(i)); }
    }

    Field2D<double, 1000, 1000> mat{};
    std::fill_n(mat.get_data(), mat.size(), 42.0);
    for (Index i = 0; i < mat.extent(0); ++i) {
      for (Index j = 0; j < mat.extent(1); ++j) {
        if (mat(i, j) != 42.0) { Igor::Warn("Expect value {} but got", 42.0, mat(i, j)); }
      }
    }
  }

  // ===============================================================================================
  {
    Field1D<double, 5, 2> vec{};

    std::iota(vec.get_data(), vec.get_data() + vec.size(), -2.0);
    for (Index i = -2; i < vec.extent(0) + 2; ++i) {
      if (std::abs(vec(i) - static_cast<double>(i)) > 1e-12) {
        Igor::Warn("Expect value {} but got", 42.0, vec(i));
      }
    }

    Field2D<double, 5, 5, 2> mat{};
    std::iota(mat.get_data(), mat.get_data() + mat.size(), 0.0);
    Index counter = 0;
    for (Index i = -2; i < mat.extent(0) + 2; ++i) {
      for (Index j = -2; j < mat.extent(1) + 2; ++j) {
        if (std::abs(mat(i, j) - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat(i, j));
        }
        counter += 1;
      }
    }

    Field2D<double, 5, 5, 2, Layout::F> mat_f{};
    std::iota(mat_f.get_data(), mat_f.get_data() + mat_f.size(), 0.0);
    counter = 0;
    for (Index j = -2; j < mat_f.extent(1) + 2; ++j) {
      for (Index i = -2; i < mat_f.extent(0) + 2; ++i) {
        if (std::abs(mat_f(i, j) - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat_f(i, j));
        }
        counter += 1;
      }
    }
  }

  // ===============================================================================================
  {
    Field1D<double, 1000, 2> vec{};
    std::iota(vec.get_data(), vec.get_data() + vec.size(), -2.0);
    for (Index i = -2; i < vec.extent(0) + 2; ++i) {
      if (std::abs(vec(i) - static_cast<double>(i)) > 1e-12) {
        Igor::Warn("Expect value {} but got {}", i, vec(i));
      }
    }

    Field2D<double, 1000, 1000, 2> mat{};
    std::iota(mat.get_data(), mat.get_data() + mat.size(), 0.0);
    Index counter = 0;
    for (Index i = -2; i < mat.extent(0) + 2; ++i) {
      for (Index j = -2; j < mat.extent(1) + 2; ++j) {
        if (std::abs(mat(i, j) - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat(i, j));
        }
        counter += 1;
      }
    }

    Field2D<double, 1000, 1000, 2, Layout::F> mat_f{};
    std::iota(mat_f.get_data(), mat_f.get_data() + mat_f.size(), 0.0);
    counter = 0;
    for (Index j = -2; j < mat_f.extent(1) + 2; ++j) {
      for (Index i = -2; i < mat_f.extent(0) + 2; ++i) {
        if (std::abs(mat_f(i, j) - static_cast<double>(counter)) > 1e-12) {
          Igor::Warn("Expect value {} but got {}", counter, mat_f(i, j));
        }
        counter += 1;
      }
    }
  }
}
