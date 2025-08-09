#include <xtensor.hpp>

template<typename T>
void Sin(const xt::xarray<T>& input, xt::xarray<T>& output){
	output = xt::sin(input);
}




