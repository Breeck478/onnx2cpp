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
#include <xtensor/xsort.hpp>
struct ArgMaxParams
{
  const int axis = 0;
  const int keepdims = 1;
  const int select_last_index = 0;
};
template <typename T>
void ArgMax(
    const xt::xarray<T> &data,
    xt::xarray<int64_t> &reduced,
    const ArgMaxParams &params = ArgMaxParams())
{
  int axis = params.axis;
  int keepdims = params.keepdims;
  int select_last_index = params.select_last_index;

  if (select_last_index == 0)
  {
    // _argmax
    xt::xarray<T> tempRes = xt::argmax(data, axis);
    if ((keepdims == 1) && (tempRes.dimension() < data.dimension()))
    {
      reduced = xt::cast<int>(xt::expand_dims(tempRes, axis));
    }
    else
    {
      reduced = xt::cast<int>(tempRes);
    }
  }
  else
  {
    // _argmax_use_numpy_select_last_index
    xt::xarray<T> tempData = xt::flip(data, axis);
    xt::xarray<T> tempRes = xt::argmax(tempData, axis);
    tempRes = tempData.shape()[axis] - tempRes - 1;
    if (keepdims == 1)
    {
      reduced = xt::cast<int>(xt::expand_dims(tempRes, axis));
    }
    else
    {
      reduced = xt::cast<int>(tempRes);
    }
  }
}
