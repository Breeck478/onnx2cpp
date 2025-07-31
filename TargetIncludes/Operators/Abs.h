#include <xtensor/xarray.hpp>


template <typename Tx, typename Tr>
void Abs(const xt::xarray<Tx> &x, xt::xarray<Tr> &r)
{
  r = xt::abs(x);
}

