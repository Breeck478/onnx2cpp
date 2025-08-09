#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
struct GemmParams
{
  const double alpha = 1.0;
  const double beta = 1.0;
  const bool transA = false;
  const bool transB = false;
};
template <typename T1, typename T2, typename T3, typename T4>
void Gemm(const xt::xarray<T1>& A, const xt::xarray<T2>& B, const xt::xarray<T3>& C, xt::xarray<T4>& Y, const GemmParams& params = GemmParams())
{
    auto alpha = params.alpha;
    auto beta = params.beta;
    auto transA = params.transA;
    auto transB = params.transB;

    xt::xarray<T1> A_proc = transA ? xt::transpose(A) : A;
    xt::xarray<T2> B_proc = transB ? xt::transpose(B) : B;
    xt::xarray<T4> dot_Res;
    if ((A_proc.shape().size() == 1) && (B_proc.shape().size() == 1))
    {
        dot_Res = xt::linalg::vdot(A_proc, B_proc);
    }
    else if ((A_proc.shape().size() > 2) || (B_proc.shape().size() > 2))
    {
        xt::xarray<T1> correctA = xt::broadcast(A_proc, B_proc.shape());

        dot_Res = xt::linalg::tensordot(correctA, B_proc);
    }
    else
    {
        dot_Res = xt::linalg::dot(A_proc, B_proc);
    }
    xt::xarray<T4> G = dot_Res * alpha;

    if (C.size() > 0 && beta != 0.0)
    {

        Y = G + (C * beta);
    }
    else {
        Y = G;
    }

}

