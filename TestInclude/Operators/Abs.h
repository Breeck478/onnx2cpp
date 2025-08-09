#include <xtensor.hpp>


template <typename Tx, typename Tr>
void Abs(const xt::xarray<Tx> &x, xt::xarray<Tr> &y)
{
  y = xt::abs(x);
}

