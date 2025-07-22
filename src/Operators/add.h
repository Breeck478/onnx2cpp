#include <xtensor/xarray.hpp>


template <typename Tx, typename Ty, typename Tr>
void Add(const xt::xarray<Tx> &x, const xt::xarray<Ty>& y, xt::xarray<Tr> &r)
{
	r = x + y;
}

