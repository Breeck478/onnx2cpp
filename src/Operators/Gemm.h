#include <cmath>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
// #include <xtensor/dotUtil.h>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xeval.hpp>
struct GemmParams
{
  const double alpha = 1.0;
  const double beta = 1.0;
  const bool transA = false;
  const bool transB = false;
};
template <typename T>
void Gemm(
    const xt::xarray<T> &A,
    const xt::xarray<T> &B,
    const xt::xarray<T> &C, 
    xt::xarray<T> &Y,
    const GemmParams &params = GemmParams())
{
  auto alpha = params.alpha;
  auto beta = params.beta;
  auto transA = params.transA;
  auto transB = params.transB;

  xt::xarray<T> A_proc = transA ? xt::transpose(A) : A;
  xt::xarray<T> B_proc = transB ? xt::transpose(B) : B;

  xt::xarray<T> dot_Res;
  if ((A_proc.shape().size() == 1) && (B_proc.shape().size() == 1))
  {

    dot_Res = xt::linalg::vdot(A_proc, B_proc);
  }
  else if ((A_proc.shape().size() > 2) || (B_proc.shape().size() > 2))
  {

    xt::xarray<T> correctA = xt::broadcast(A_proc, B_proc.shape());

    dot_Res = xt::linalg::tensordot(correctA, B_proc);
  }
  else
  {
      dot_Res = xt::linalg::dot(A_proc, B_proc);

      xt::xarray<T> G = alpha * dot_Res;

      if (C.size() > 0 && beta != 0.0)
      {

          Y = G + (beta * C);
      }
      else {
          Y = G;
      }

  }

