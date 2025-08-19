#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
struct ArgMaxParams
{
  const int axis = 0;
  const int keepdims = 1;
  const int select_last_index = 0;
};
template <typename T>
void ArgMax(const xt::xarray<T>& data, xt::xarray<int64_t>& reduced, const ArgMaxParams& params = ArgMaxParams())
{
    int axis = params.axis < 0 ? (data.dimension() + params.axis) : params.axis;
    int keepdims = params.keepdims;
    int select_last_index = params.select_last_index;

    if (select_last_index == 0)
    {
        // _argmax
        xt::xarray<T> tempRes = xt::argmax(data, axis);
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
        // _argmax_use_numpy_select_last_index
        xt::xarray<T> tempData = xt::flip(data, axis);
        xt::xarray<T> tempRes = xt::argmax(tempData, axis);
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
