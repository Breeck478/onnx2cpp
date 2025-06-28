#include <xtensor/xarray.hpp>


template <typename T>
void Pow(const xt::xarray<T> &x, const xt::xarray<T> &y, xt::xarray<T> &r)
{
  r = xt::power(x, y);
}

