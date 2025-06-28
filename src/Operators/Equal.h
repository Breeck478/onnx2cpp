#include <xtensor/xarray.hpp>

template <typename T>
void Equal(const xt::xarray<T> &x, const xt::xarray<T> &y, xt::xarray<bool> &r)
{
  r = xt::equal(x, y);
} 
