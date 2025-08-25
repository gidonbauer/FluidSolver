#include <algorithm>
#include <random>

#include <Igor/Logging.hpp>
#include <Igor/Timer.hpp>

#define FS_PARALLEL_THRESHOLD 1000
#define FS_INDEX_TYPE std::int64_t

#include "Container.hpp"
#include "ForEach.hpp"

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

// -------------------------------------------------------------------------------------------------
constexpr Float TOL = 1e-8;

template <typename Float>
auto is_equal(Float a, Float b) -> bool {
  return std::abs(a - b) < TOL;
}

template <typename Float, Index N, Index NGHOST>
auto is_equal(const Vector<Float, N, NGHOST>& a, const Vector<Float, N, NGHOST>& b) -> bool {
  bool equal = true;
  for_each_i(a, [&](Index i) { equal = std::abs(a(i) - b(i)) < TOL; });
  return equal;
}

template <typename Float, Index M, Index N, Index NGHOST, Layout LAYOUT>
auto is_equal(const Matrix<Float, M, N, NGHOST, LAYOUT>& A,
              const Matrix<Float, M, N, NGHOST, LAYOUT>& B) -> bool {
  bool equal = true;
  for_each_i(A, [&](Index i, Index j) { equal = std::abs(A(i, j) - B(i, j)) < TOL; });
  return equal;
}

// -------------------------------------------------------------------------------------------------
auto vecadd() -> bool {
  Vector<Float, M, 0> A{};
  std::generate_n(A.get_data(), A.size(), rand_float);
  Vector<Float, M> B{};
  std::generate_n(B.get_data(), B.size(), rand_float);

  Vector<Float, M, 0> C_ref{};
  IGOR_TIME_SCOPE("Vector addition: Reference solution") {
    for_each_i(C_ref, [&](Index i) { C_ref(i) = A(i) + B(i); });
  }

  Vector<Float, M, 0> C{};
  IGOR_TIME_SCOPE("Vector addition: GPU solution") {
    for_each_i<Exec::Parallel>(
        C, [A = A.view(), B = B.view(), C = C.view()](Index i) mutable { C(i) = A(i) + B(i); });
  }

  return is_equal(C, C_ref);
}

// -------------------------------------------------------------------------------------------------
auto matmul() -> bool {
  Matrix<Float, M, K, 0> A{};
  std::generate_n(A.get_data(), A.size(), rand_float);
  Matrix<Float, K, N, 0, Layout::F> B{};
  std::generate_n(B.get_data(), B.size(), rand_float);

  Matrix<Float, M, N, 0> C_ref{};
  IGOR_TIME_SCOPE("Matrix multiplication: Reference solution") {
    for_each_i(C_ref, [&](Index i, Index j) {
      for (Index k = 0; k < K; ++k) {
        C_ref(i, j) += A(i, k) * B(k, j);
      }
    });
  }

  Matrix<Float, M, N, 0> C{};
  IGOR_TIME_SCOPE("Matrix multiplication: GPU solution") {
    for_each_i<Exec::Parallel>(
        C, [A = A.view(), B = B.view(), C = C.view()](Index i, Index j) mutable {
          for (Index k = 0; k < K; ++k) {
            C(i, j) += A(i, k) * B(k, j);
          }
        });
  }

  return is_equal(C, C_ref);
}

// -------------------------------------------------------------------------------------------------
auto dotprod() -> bool {
  Vector<Float, M, 0> A{};
  std::generate_n(A.get_data(), A.size(), rand_float);
  Vector<Float, M> B{};
  std::generate_n(B.get_data(), B.size(), rand_float);

  Float C_ref = 0.0;
  IGOR_TIME_SCOPE("Dot-product: Reference solution") {
    for_each_i(A, [&](Index i) { C_ref += A(i) * B(i); });
  }

  std::atomic<Float> C = 0.0;
  IGOR_TIME_SCOPE("Dot-product: GPU solution") {
    for_each_i<Exec::Parallel>(A, [A = A.view(), B = B.view(), &C](Index i) { C += A(i) * B(i); });
  }

  return is_equal(static_cast<Float>(C), C_ref);
}

// -------------------------------------------------------------------------------------------------
auto main() -> int {
  const auto vecadd_correct = vecadd();
  if (!vecadd_correct) { Igor::Warn("Vector addition failed."); }

  const auto matmul_correct = matmul();
  if (!matmul_correct) { Igor::Warn("Matrix multiplication failed."); }

  const auto dotprod_correct = dotprod();
  if (!dotprod_correct) { Igor::Warn("Dot-product failed."); }

  return (vecadd_correct && matmul_correct && dotprod_correct) ? 0 : 1;
}
