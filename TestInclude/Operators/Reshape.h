#include <xtensor.hpp>

//#include <xtensor/xoperation.hpp>
struct ReshapeParams
{
	const int allowzero = 0;
};
template<typename T>
void Reshape(const xt::xarray<T> &data, const xt::xarray<int64_t>& shape, xt::xarray<T>& reshaped, const ReshapeParams& params = ReshapeParams()) {
	//typename xt::xarray<T>::shape_type  newShape;
	//for (size_t i = 0; i < shape.size(); ++i) {
	//	newShape[i] = shape[i];
	//}

	std::vector<int64_t> newShape(shape.size());
	for (size_t i = 0; i < shape.size(); ++i) {
		if (shape(i) == 0) {
			std::cout << "Warning: Reshape encountered a zero dimension in the shape array. This may lead to unexpected behavior because XTensor does not support 0 Shape (See xcontainer.compute_size)." << std::endl;
		}
			newShape[i] = shape(i);

	}
	if (params.allowzero == 0) {
		auto dataShape = data.shape();
		xt::xarray<bool> zerosIndex = xt::equal(shape, 0);
		for (size_t i = 0; i < zerosIndex.size(); ++i) {
			if(zerosIndex[i]){
				int64_t dim_i = dataShape[i];
				newShape[i] = dim_i; // Replace zero with the corresponding dimension size from the original data
			}
		}
	}
	//if (data.shape().size() != newShape.size()) {
	//	throw std::runtime_error("Shape size mismatch: input and output shape must have the same number of elements."); // specified by onnx
	//}
	auto dataCopy = data; // Copy the data to avoid modifying the original array
	reshaped = dataCopy.reshape(newShape);
}



