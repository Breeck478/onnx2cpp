#include <xtensor.hpp>
struct SoftmaxParams
{
	const int axis = -1;
};

template <typename T>
void Softmax(const xt::xarray<T>& input, xt::xarray<T>& output, const SoftmaxParams& params = SoftmaxParams()) {
	int axis = params.axis < 0 ? (input.dimension() + params.axis) : params.axis;
	typename xt::xarray<T>::shape_type keepdimsShape = {};
	for (size_t i = 0; i < input.shape().size(); i++) {
#
		if (i != axis)
			keepdimsShape.push_back(input.shape()[i]);
		else
			keepdimsShape.push_back(1);
	};
	if (input.size() == 0) {
		output = input;
		return;
	}
	// manualy reshape tensors to copy keepdims behavior from numpy
	xt::xarray<T> keepdimsTmp = xt::amax(input, { axis }); //xt::keep_dims_type
	keepdimsTmp.reshape(keepdimsShape);
	xt::xarray<T> tmp = input - keepdimsTmp;
	output = xt::exp(tmp);
	xt::xarray<T> keepdimsOutput = xt::sum(output, { axis });
	keepdimsOutput.reshape(keepdimsShape);
	output = output / keepdimsOutput;
}