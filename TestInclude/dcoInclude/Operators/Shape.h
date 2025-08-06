#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>


struct ShapeParams
{
	const int end = INFINITY;
	const int start = 0;
};
template <typename Tx, typename Tr>
void Shape(const xt::xarray<Tx>& x, xt::xarray<Tr>& r, const ShapeParams& params = ShapeParams())
{
	int start = params.start;
	int end = params.end;
	if (abs(start) >= x.shape().size() || abs(end) > x.shape().size()) {
		throw std::runtime_error("ShapeParams are out of bounds");
	}
	xt::xarray<Tr> shape = xt::adapt(x.shape(), { x.shape().size() - start - end }, xt::no_ownership(), x.shape().data() + start);
}

