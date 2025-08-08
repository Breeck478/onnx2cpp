#include <xtensor.hpp>

template <typename Tx, typename Ty, typename Tr>
void MatMul(const xt::xarray<Tx> x, const xt::xarray<Ty> y, xt::xarray<Tr> &r)
{
  r = x * y;
}
