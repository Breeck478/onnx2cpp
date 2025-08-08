#include <xtensor.hpp>


template <typename T>
void Sub(const xt::xarray<T> &x, const xt::xarray<T> &y, xt::xarray<T> &r)
{
  r = x - y;
}

