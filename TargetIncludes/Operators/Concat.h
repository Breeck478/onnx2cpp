#include <xtensor/xarray.hpp>
#include <tuple>
struct ConcatParams
{
	const int axis = 1;
};
template <typename T>
void Concat(const std::vector<xt::xarray<T>> x, xt::xarray<T> &r, ConcatParams params = ConcatParams())
{
	auto x_tuple = std::make_tuple(x.begin(), x.end());
	r = xt::concatenate(x_tuple, params.axis);
} 
