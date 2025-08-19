#include <xtensor.hpp>

//#include <xtensor/xoperation.hpp>
struct ReshapeParams
{
	const int allowzero = 0;
};
template<typename T>
void Reshape(const xt::xarray<T> &data, const xt::xarray<int64_t>& shape, xt::xarray<T>& reshaped, const ReshapeParams& params = ReshapeParams()) {
	bool containsZero = false;
	std::vector<int64_t> newShape(shape.size());
	for (size_t i = 0; i < shape.size(); ++i) {
		if (shape(i) == 0) {
			containsZero = true;
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
	else if (containsZero){ // if allowzero is True and the shape contains a zero. Data is lost. Therefor just initialise a new Array with this shape because reshape does not work with 0
		reshaped = xt::zeros<T>(newShape);
		return;
	}
	auto dataCopy = data; // Copy the data to avoid modifying the original array
	reshaped = dataCopy.reshape(newShape);
}



