#include <xtensor/xarray.hpp>

template <typename T>
void MatMul(const xt::xarray<T> &x, const xt::xarray<T> &y, xt::xarray<T> &r)
{
  r = x * y;
}
