#include <vector>

#include <Accelerate/Accelerate.h>

#include <Igor/Logging.hpp>

auto main() -> int {
  std::vector<int> row       = {0, 1, 3, 1, 2, 3, 2, 3};
  std::vector<int> column    = {0, 0, 0, 1, 1, 1, 2, 3};
  std::vector<double> values = {10.0, 1.0, 2.5, 12.0, -0.3, 1.1, 9.5, 6.0};

  SparseAttributes_t attributes{};
  const int nrows             = 4;
  const int ncols             = 4;
  const int nblocks           = 8;
  const int blocksize         = 1;
  const SparseMatrix_Double A = SparseConvertFromCoordinate(
      nrows, ncols, nblocks, blocksize, attributes, row.data(), column.data(), values.data());

  std::vector<double> x_values = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> b_values = {1.0, 2.0, 3.0, 4.0};

  DenseVector_Double x{
      .count = static_cast<int>(x_values.size()),
      .data  = x_values.data(),
  };
  DenseVector_Double b{
      .count = static_cast<int>(b_values.size()),
      .data  = b_values.data(),
  };

  SparseCGOptions opts{
      .reportError   = [](const char* message) { std::cerr << "ERROR: " << message; },
      .maxIterations = 50,
      .atol          = 1e-6,
      .rtol          = 0.0,
      .reportStatus  = [](const char* message) { std::cout << message; },
  };
  const auto status =
      SparseSolve(SparseConjugateGradient(opts), A, b, x, SparsePreconditionerDiagScaling);

  switch (status) {
    case SparseIterativeConverged:      Igor::Info("OK!"); break;
    case SparseIterativeIllConditioned: Igor::Warn("Ill conditioned."); return 1;
    case SparseIterativeInternalError:  Igor::Warn("Internal error."); return 1;
    case SparseIterativeMaxIterations:  Igor::Warn("Max. iterations."); return 1;
    case SparseIterativeParameterError: Igor::Warn("Parameter error."); return 1;
  }

  Igor::Info("x = {}", x_values);
}
