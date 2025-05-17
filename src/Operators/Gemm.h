#include <cmath>
#include <vector>

template <typename T>
void Gemm(const std::vector<std::vector<T>> &A,
         const std::vector<std::vector<T>> &B,
         std::vector<std::vector<T>> &C,
         float alpha = 1.0f,
         float beta = 0.0f)
{
  size_t M = A.size();
  size_t N = B[0].size();
  size_t K = A[0].size();

  // Initialize result matrix if empty
  if (C.size() != M || C[0].size() != N)
    C = std::vector<std::vector<T>>(M, std::vector<T>(N, 0.0f));

  for (size_t i = 0; i < M; ++i)
  {
    for (size_t j = 0; j < N; ++j)
    {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k)
      {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = alpha * sum + beta * C[i][j];
    }
  }
}
