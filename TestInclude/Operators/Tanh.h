#include <xtensor.hpp>

template<typename T>
void Tanh(const xt::xarray<T>& input, xt::xarray<T>& output) {
	output = xt::tanh(input);
}




