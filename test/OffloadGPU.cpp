#include <algorithm>
#include <atomic>
#include <random>

#include <omp.h>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#define FS_PARALLEL_THRESHOLD 1000

#include "Container.hpp"
#include "ForEach.hpp"
#include "Utility.hpp"

using Float       = double;
constexpr Index M = 1024;
constexpr Index K = 512;
constexpr Index N = 1024;

// -------------------------------------------------------------------------------------------------
auto rand_float() -> Float {
  static std::mt19937 rng(std::random_device{}());
  static std::normal_distribution<Float> dist(0.0, 10.0);
  return dist(rng);
}

auto rand_index(Index lo, Index hi) -> Index {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<Index> dist(lo, hi);
  return dist(rng);
}

// -------------------------------------------------------------------------------------------------
constexpr Float TOL = 1e-8;

template <typename Float>
auto is_equal(Float a, Float b) -> bool {
  return std::abs(a - b) < TOL;
}

template <typename Float, Index N, Index NGHOST>
auto is_equal(const Field1D<Float, N, NGHOST>& a, const Field1D<Float, N, NGHOST>& b) -> bool {
  bool equal = true;
  for_each_i(a, [&](Index i) { equal = std::abs(a(i) - b(i)) < TOL; });
  return equal;
}

template <typename Float, Index M, Index N, Index NGHOST, Layout LAYOUT>
auto is_equal(const Field2D<Float, M, N, NGHOST, LAYOUT>& A,
              const Field2D<Float, M, N, NGHOST, LAYOUT>& B) -> bool {
  bool equal = true;
  for_each_i(A, [&](Index i, Index j) { equal = std::abs(A(i, j) - B(i, j)) < TOL; });
  return equal;
}

// -------------------------------------------------------------------------------------------------
auto vecadd() -> bool {
  struct {
    Field1D<Float, M, 0> A{};
    Field1D<Float, M, 0> B{};
    Field1D<Float, M, 0> C{};
    Field1D<Float, M, 0> C_ref{};
  } data{};
  std::generate_n(data.A.get_data(), data.A.size(), rand_float);
  std::generate_n(data.B.get_data(), data.B.size(), rand_float);

  IGOR_TIME_SCOPE("Field1D addition: CPU solution") {
    for_each_i<Exec::Parallel>(data.C_ref, [&](Index i) { data.C_ref(i) = data.A(i) + data.B(i); });
  }

  IGOR_TIME_SCOPE("Field1D addition: GPU solution") {
    for_each_i<Exec::ParallelGPU>(data.C, [&](Index i) { data.C(i) = data.A(i) + data.B(i); });
  }

  return is_equal(data.C, data.C_ref);
}

// -------------------------------------------------------------------------------------------------
auto matmul() -> bool {
  struct {
    Field2D<Float, M, K, 0> A{};
    Field2D<Float, K, N, 0, Layout::F> B{};
    Field2D<Float, M, N, 0> C{};
    Field2D<Float, M, N, 0> C_ref{};
  } data{};
  std::generate_n(data.A.get_data(), data.A.size(), rand_float);
  std::generate_n(data.B.get_data(), data.B.size(), rand_float);

  IGOR_TIME_SCOPE("Field2D multiplication: CPU solution") {
    for_each_i<Exec::Parallel>(data.C_ref, [&](Index i, Index j) {
      for (Index k = 0; k < K; ++k) {
        data.C_ref(i, j) += data.A(i, k) * data.B(k, j);
      }
    });
  }

  IGOR_TIME_SCOPE("Field2D multiplication: GPU solution") {
    for_each_i<Exec::ParallelGPU>(data.C, [&](Index i, Index j) {
      for (Index k = 0; k < K; ++k) {
        data.C(i, j) += data.A(i, k) * data.B(k, j);
      }
    });
  }

  return is_equal(data.C, data.C_ref);
}

// -------------------------------------------------------------------------------------------------
auto dotprod() -> bool {
  struct {
    Field1D<Float, M, 0> A{};
    Field1D<Float, M> B{};
    std::atomic<Float> C     = 0.0;
    std::atomic<Float> C_ref = 0.0;
  } data{};
  std::generate_n(data.A.get_data(), data.A.size(), rand_float);
  std::generate_n(data.B.get_data(), data.B.size(), rand_float);

  IGOR_TIME_SCOPE("Dot-product: CPU solution") {
    for_each_i<Exec::Parallel>(data.A, [&](Index i) { data.C_ref += data.A(i) * data.B(i); });
  }

  IGOR_TIME_SCOPE("Dot-product: GPU solution") {
    for_each_i<Exec::ParallelGPU>(data.A, [&](Index i) { data.C += data.A(i) * data.B(i); });
  }

  return is_equal(static_cast<Float>(data.C), static_cast<Float>(data.C_ref));
}

// -------------------------------------------------------------------------------------------------
auto max_reduce() -> bool {
  Field2D<Float, M, N, 0> A{};
  std::generate_n(A.get_data(), A.size(), rand_float);

  constexpr Float MAX_VALUE             = 4.2e12;
  A(rand_index(0, M), rand_index(0, N)) = MAX_VALUE;

  Float C_ref1                          = 0.0;
  IGOR_TIME_SCOPE("Max-reduce: CPU solution (OpenMP critical)") {
    for_each_i<Exec::Parallel>(A, [&](Index i, Index j) {
#pragma omp critical
      { C_ref1 = std::max(C_ref1, A(i, j)); }
    });
  }

  std::atomic<Float> C_ref2 = 0.0;
  IGOR_TIME_SCOPE("Max-reduce: CPU solution (C++ atomic)") {
    for_each_i<Exec::Parallel>(A,
                               [&](Index i, Index j) { update_maximum_atomic(C_ref2, A(i, j)); });
  }

  std::atomic<Float> C = 0.0;
  IGOR_TIME_SCOPE("Max-reduce: GPU solution (C++ atomic)") {
    for_each_i<Exec::ParallelGPU>(A, [&](Index i, Index j) { update_maximum_atomic(C, A(i, j)); });
  }

  return is_equal(static_cast<Float>(C), C_ref1) &&
         is_equal(static_cast<Float>(C), static_cast<Float>(C_ref2)) &&
         is_equal(static_cast<Float>(C), MAX_VALUE);
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  omp_set_num_threads(4);

  std::array funcs = {
      &vecadd,
      &matmul,
      &dotprod,
      &max_reduce,
  };
  std::array names = {
      "Field1D addition",
      "Field2D multiplication",
      "Dot-product",
      "Max-reduce",
  };
  static_assert(funcs.size() == names.size());

  bool any_failed = false;
  for (size_t i = 0; i < funcs.size(); ++i) {
    const bool correct = funcs[i]();
    if (!correct) {
      Igor::Warn("{} failed.", names[i]);
      any_failed = true;
    }
  }

  return any_failed ? 1 : 0;
}
