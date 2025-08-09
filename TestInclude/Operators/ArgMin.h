#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>

struct ArgMinParams
{
  const int axis = 0;
  const int keepdims = 1;
  const int select_last_index = 0;
};
template <typename T>
void ArgMin(
    const xt::xarray<T> &data,
    xt::xarray<int64_t> &reduced,
    const ArgMinParams &params = ArgMinParams())
{
  int axis = params.axis;
  int keepdims = params.keepdims;
  int select_last_index = params.select_last_index;

  if (select_last_index == 0)
  {
    // _argmin
    xt::xarray<T> tempRes = xt::argmin(data, axis);
    if ((keepdims == 1) && (tempRes.dimension() < data.dimension()))
    {
      reduced = xt::cast<int64_t>(xt::expand_dims(tempRes, axis));
    }
    else
    {
      reduced = xt::cast<int64_t>(tempRes);
    }
  }
  else
  {
    // _argmin_use_numpy_select_last_index
    xt::xarray<T> tempData = xt::flip(data, axis);
    xt::xarray<T> tempRes = xt::argmin(tempData, axis);
    tempRes = tempData.shape()[axis] - tempRes - 1;
    if (keepdims == 1)
    {
      reduced = xt::cast<int64_t>(xt::expand_dims(tempRes, axis));
    }
    else
    {
      reduced = xt::cast<int64_t>(tempRes);
    }
  }
}
