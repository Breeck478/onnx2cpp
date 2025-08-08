#include <xtensor.hpp>

template <typename Tx, typename Ty>
void Greater(const xt::xarray<Tx> &x, const xt::xarray<Ty> &y, xt::xarray<bool> &r)
{
  r = xt::greater(x, y);
} 
