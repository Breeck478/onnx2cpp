#include <xtensor.hpp>
#include <tuple>
#include <optional>
template <typename T>
xt::xarray<T> appendValue(const xt::xarray<T>& arr, T val)
{
    xt::xarray<T> new_arr = xt::empty<T>({ arr.size() + 1 });
    std::copy(arr.begin(), arr.end(), new_arr.begin());
    new_arr(arr.size()) = val; // letzten Wert setzen
    return new_arr;
}

struct ConcatParams
{
	const std::optional<int> axis = 1;
};
template <typename T>
void Concat(const std::vector<xt::xarray<T>> x, xt::xarray<T> &r, ConcatParams params = ConcatParams())
{   
	//Check if axis is set because it is required by onnx
	if (!params.axis.has_value()) {
		throw std::runtime_error("Axis must be set for concatenation.");
	}
	//std::tuple<int> x_tuple = std::make_tuple(1);//std::make_tuple(x.begin(), x.end());
	r = x[0];//xt::concatenate(x_tuple, params.axis);
} 
template <typename T>
