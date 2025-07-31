#include <xtensor/xarray.hpp>

template<typename T>
void Identity(const xt::xarray<T>& x, xt::xarray<T>& r) {
	// Identity operation simply copies the input array to the output array
	// It is not allowed that the input array is empty
	if (x.size() == 0) {
		throw std::runtime_error("Identity operation requires a non-empty input array.");
	}
	r = x;
}



