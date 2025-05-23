#include <cmath>
#include <vector>
#include <stdexcept>
#include <cassert>

template <typename T>
void gemm(
    const std::vector<T> &A, int M, int K,
    const std::vector<T> &B,  int Kb, int N,
    const std::vector<T> &C,
    const float alpha = 1.0f,
    const float beta = 1.0f,
    const bool transA = false,
    const bool transB = false,
    std::vector<T> &Y)
{
  // Helper to transpose matrix
  if (!transA && K != Kb)
  {
    throw std::invalid_argument("Inkompatible Dimensionen für A und B.");
  }
  if (transA && M != Kb)
  {
    throw std::invalid_argument("Inkompatible Dimensionen bei transA.");
  }

  int a_rows = transA ? K : M;
  int a_cols = transA ? M : K;
  int b_rows = transB ? N : K;
  int b_cols = transB ? K : N;

  if (a_cols != b_rows)
  {
    throw std::invalid_argument("Inkompatible Matrixmaße nach Transponieren.");
  }

  Y.resize(M * N, 0.0f);

  // Matrix-Multiplikation: alpha * A * B
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k)
      {
        float a_val = transA ? A[k * M + i] : A[i * K + k];
        float b_val = transB ? B[j * K + k] : B[k * N + j];
        sum += a_val * b_val;
      }
      Y[i * N + j] = alpha * sum;
    }
  }

  // Optional: beta * C hinzufügen
  if (!C.empty())
  {
    if (C.size() != static_cast<size_t>(M * N))
    {
      throw std::invalid_argument("C hat falsche Dimension.");
    }
    for (int i = 0; i < M * N; ++i)
    {
      Y[i] += beta * C[i];
    }
  }
}
