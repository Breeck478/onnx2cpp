#include <xtensor/xarray.hpp>

template<typename T>
void Relu(const xt::xarray<T>& x, xt::xarray<T>& r) {
	r = xt::maximum(x, 0);
}



