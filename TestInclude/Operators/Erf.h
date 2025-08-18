#include <xtensor.hpp>

template <typename T>
void Erf(const xt::xarray<T> &input, xt::xarray<T> &output)
{
	output = xt::erf(input);
} 