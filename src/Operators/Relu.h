#include <xtensor/xarray.hpp>

template<typename T>
void Identity(const xt::xarray<T>& x, xt::xarray<T>& r) {
	
	r = x;
}



