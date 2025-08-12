#include <xtensor.hpp>


template <typename Ta, typename Tb, typename Tc>
void Sub(const xt::xarray<Ta> &a, const xt::xarray<Tb> &b, xt::xarray<Tc> &c)
{
  c = a - b;
}

