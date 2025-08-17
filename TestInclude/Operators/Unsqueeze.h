#include <xtensor.hpp>

struct UnsqueezeParams
{
	const xt::xarray<int64_t> axes = {};
};



template <typename T>
void Unsqueeze(const xt::xarray<T> &data, const xt::xarray<int64_t> &axis, xt::xarray<T> &expanded)
{
	xt::xarray<int64_t> sortedAxis = xt::sort(axis, 1); // axis should be a 1D array of integers
	expanded = data;
	int axis_size = expanded.shape().size();
	for (size_t i = 0; i < sortedAxis.size(); ++i) {
		int val = sortedAxis[i];
		if (val < 0) {
			val = axis_size + val;
		}
		expanded = xt::expand_dims(expanded, val);
		axis_size = expanded.shape().size();

  }
}

template <typename T>
void Unsqueeze(const xt::xarray<T>& data, xt::xarray<T>& expanded, const UnsqueezeParams& params = UnsqueezeParams()) {
	Unsqueeze(data, params.axes, expanded);
}
