#include <xtensor.hpp>
#include <xtensor/xbuilder.hpp>
#include <tuple>
struct ConcatParams
{
	const int axis = 1;
};
template <typename T>
void Concat(const std::vector<xt::xarray<T>> x, xt::xarray<T>& r, ConcatParams params = ConcatParams())
{
	for (const xt::xarray<T>& arr : x) {
		r = xt::concatenate(xt::xtuple(arr, r), params.axis);
	}	
}
