#include <xtensor/xarray.hpp>
#include <iostream>

template <typename Tx, typename Ty, typename Tr>
void Where(const xt::xarray<bool>& condition, const xt::xarray<Tx>& x, const xt::xarray<Ty>& y, xt::xarray<Tr>& r)
{ // Ensure that x and y are of the same type
	static_assert(std::is_same<Tx, Ty>::value, "Types of x and y must be the same for Where operation");
	r = xt::where(condition, x, y);
}
