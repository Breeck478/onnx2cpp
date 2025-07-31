#include <xtensor/xarray.hpp>
#include <tuple>
struct ConcatParams
{
	const int axis = 1;
};
template <typename T>
void Concat(const std::tuple<xt::xarray<T>> &x, xt::xarray<T> &r, ConcatParams params = ConcatParams())
{
  r = xt::concatenate(x, params.axis);
} 
