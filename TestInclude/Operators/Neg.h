#include <xtensor.hpp>

template <typename T>
void Neg(const xt::xarray<T> &x, xt::xarray<T> &y)
{
	y = x * -1;
} 