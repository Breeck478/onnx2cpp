#include <xtensor.hpp>


template <typename T>
void Sub(const xt::xarray<T> &a, const xt::xarray<T> &b, xt::xarray<T> &c)
{
  c = a - b;
}

