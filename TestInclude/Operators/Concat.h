#include <xtensor.hpp>
#include <tuple>
#include <optional>
struct ConcatParams
{
	const std::optional<int> axis = 1;
};
template <typename T>
void Concat(const std::vector<xt::xarray<T>> x, xt::xarray<T>& r, ConcatParams params = ConcatParams())
{
	//Check if axis is set because it is required by onnx
	if (!params.axis.has_value()) {
		throw std::runtime_error("Axis must be set for concatenation.");
	}
	if (x.empty()) {
		throw std::runtime_error("Input vector for concatenation is empty.");
	}
	if (x.size() == 1) {
		r = x[0]; // If only one array, just assign it
		return;
	}
	std::size_t rank = x[0].shape().size();
	std::size_t axis = params.axis.value() < 0 ? (rank + params.axis.value()) : params.axis.value(); // convert negative axis to positive
	if (rank <= axis) {
		throw std::runtime_error("Axis out of bounds for concatenation.");
	}
	for (std::size_t i = 1; i < x.size(); i++) {
		if (x[i].shape().size() != rank) {
			throw std::runtime_error("All input arrays must have the same rank for concatenation.");
		}
	}
	for (std::size_t i = 0; i < rank; i++) {
		size_t dim_size = x[0].shape()[i];
		for (const auto& arr : x) {
			if (i == axis) continue; // Skip the axis we are concatenating along. The dimension may differ.
			if (arr.shape()[i] != dim_size) {
				throw std::runtime_error("All input arrays must have the same shape except for the concatenation axis.");
			}
		}
	}
	
	std::vector<std::size_t> newForm;
	for (std::size_t i = 0; i < rank; i++) {
		if (i == axis) {
			std::size_t total_size = 0;
			for (const auto& arr : x) {
				total_size += arr.shape()[i];
			}
			newForm.push_back(total_size);
		}
		else {
			newForm.push_back(x[0].shape()[i]);
		}
	}
	r = xt::empty<T>(newForm); // Initialize the result array with the new shape
	std::size_t offset = 0;
	for (const auto& t : x)
	{
		xt::xstrided_slice_vector slice_vector;
		for (std::size_t dim = 0; dim < rank; ++dim)
		{
			if (dim == axis)
			{
				slice_vector.push_back(xt::range(offset, offset + t.shape()[dim]));
			}
			else
			{
				slice_vector.push_back(xt::all());
			}
		}

		auto resultStride = xt::strided_view(r, slice_vector);
		xt::noalias(resultStride) = t;
		offset += t.shape()[axis];
	}
}