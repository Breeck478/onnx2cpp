#include <xtensor.hpp>
#include <iostream>

template <typename T>
void Where(const xt::xarray<bool>& condition, const xt::xarray<T>& x, const xt::xarray<T>& y, xt::xarray<T>& output)
{ 
	output = xt::where(condition, x, y);
}
