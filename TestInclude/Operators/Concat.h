#include <xtensor.hpp>
#include <tuple>
struct ConcatParams
{
	const int axis = 1;
};
template <typename T>
void Concat(const std::vector<xt::xarray<T>> x, xt::xarray<T> &r, ConcatParams params = ConcatParams())
{
	//std::tuple<int> x_tuple = std::make_tuple(1);//std::make_tuple(x.begin(), x.end());
	r = x[0];//xt::concatenate(x_tuple, params.axis);
} 
