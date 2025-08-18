#include <xtensor.hpp>

template <typename T>
void Expand(const xt::xarray<T> &input, const xt::xarray<int64_t> &shape, xt::xarray<T> &output)
{
	typename xt::xarray<T>::shape_type shapeType = {};
	for (const auto &dim : shape) {
		shapeType.push_back(static_cast<typename xt::xarray<T>::size_type>(dim));
	}
	auto ones = xt::ones<int64_t>(shapeType);
	output = input * ones;
}
