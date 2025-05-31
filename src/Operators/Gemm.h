#include <cmath>
#include <vector>
#include <stdexcept>
#include <cassert>  
#include <iostream>
struct GemmParams {
  const float alpha = 1.0f;
  const float beta = 1.0f;
  const bool transA = false;
  const bool transB = false;
};
template <typename T>
void Gemm(const std::vector<std::vector<T>> A, const std::vector<std::vector<T>> B, const std::vector<T> C = nullptr, std::vector<std::vector<T>> &Y = nullptr, GemmParams params = nullptr)
{


  auto alpha = params.alpha;
  auto beta = params.beta;
  auto transA = params.transA;
  auto transB = params.transB;

  std::cout << "Gemm"<< std::endl;
  int K = A[0].size(); 
  int Kb = B[0].size();
  int M = A.size();
  int N = B.size();
  


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

if (a_cols != b_cols)
  {
    throw std::invalid_argument("Inkompatible Matrixmaße nach Transponieren: " + std::to_string(a_cols) + " vs. " + std::to_string(b_rows));
  }

  Y.resize(a_rows, std::vector<T>(b_rows, T(0)));

  // Matrix-Multiplikation: alpha * A * B
  for (int i = 0; i < a_rows; ++i)
  {
    for (int j = 0; j < b_rows; ++j)
    {
      T sum = 0;
      for (int k = 0; k < a_cols; ++k)
      {
        T a_val = transA ? A[k][i] : A[i][k];
        T b_val = transB ? B[j][k] : B[k][j];
        sum += a_val * b_val;
        
      }
      Y[i][j] = alpha * sum;
    }
  }


  // Optional: beta * C hinzufügen
  if (!C.empty())
  {
    std::cout << "C wird genutzt" << std::endl;
    if (C.size() != static_cast<size_t>(M * N))
    {
      throw std::invalid_argument("C hat falsche Dimension.");
    }
    for (int i = 0; i < M; ++i)
    {
      for (int j = 0; j < N; j++)
      {
      Y[i][j] += beta * C[j];
      }
    }
  }
}
