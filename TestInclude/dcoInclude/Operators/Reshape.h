#include <xtensor.hpp>

template<typename T>
void Reshape(const xt::xarray<T>& x, , const xt::xarray<T>& y, xt::xarray<T>& r) {
	// Reshape x to the shape specified by y
// Note: y should be a 1D array containing the new shape dimensions
	if (y.size() == 0) {
		throw std::invalid_argument("Shape array must not be empty.");
	}

	// Ensure that the total number of elements matches
	if (xt::size(x) != xt::prod(y)) {
		throw std::invalid_argument("Total number of elements must match for reshape.");
	}
	r = x.reshape(y);
}



