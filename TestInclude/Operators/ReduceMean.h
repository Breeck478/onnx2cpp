#include <xtensor.hpp>

//#include <xtensor/xoperation.hpp>
struct ReduceMeanParams
{
	const int keepdims = 1;
	const int noop_with_empty_axes = 0;
};
template<typename T>
void ReduceMean(const xt::xarray<T>& data, const xt::xarray<int64_t>& axes, xt::xarray<T>& reduced, const ReduceMeanParams& params = ReduceMeanParams()) {
	if (axes.size() == 0 && params.noop_with_empty_axes) {
		reduced = data;
		return;
	}
	std::vector<size_t> absAxes;
	for (size_t i = 0; i < axes.size(); i++) {
		if (axes.flat(i) < 0)
			absAxes.push_back(axes.flat(i) + data.shape().size());
		else
			absAxes.push_back(axes.flat(i));
	}
	if (!params.keepdims) {
		if (axes.size() > 0)
			reduced = xt::mean(data, absAxes);
		else
			reduced = xt::mean(data);
		return;
	}


	typename xt::xarray<T>::shape_type keepdimsShape = {};
	for (size_t i = 0; i < data.shape().size(); i++) {
#
		if (absAxes.size() > 0 && std::find(absAxes.begin(), absAxes.end(), i) == absAxes.end()) { // if axes is empty all dimension of output tneosr are 1
			keepdimsShape.push_back(data.shape()[i]);
		}
		else {
			keepdimsShape.push_back(1);
		}

	}

	if (axes.size() > 0)
		reduced = xt::mean(data, absAxes);
	else
		reduced = xt::mean(data);
	reduced.reshape(keepdimsShape);
};

template<typename T>
void ReduceMean(const xt::xarray<T>& data, xt::xarray<T>& reduced, const ReduceMeanParams& params = ReduceMeanParams()) {
	xt::xarray<int64_t> emptyAxes = xt::empty<int64_t>({0});
	ReduceMean(data, emptyAxes, reduced, params);
}

