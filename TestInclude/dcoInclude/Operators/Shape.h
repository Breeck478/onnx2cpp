#include <xtensor.hpp>
#include <limits>
//#include <xtensor/xadapt.hpp>


struct ShapeParams
{
	const int end = std::numeric_limits<int>::max();
	const int start = 0;
};
template <typename Tx, typename Tr>
void Shape(const xt::xarray<Tx>& x, xt::xarray<Tr>& r, const ShapeParams& params = ShapeParams())
{
	//int start = params.start;
	//int end = params.end;
	//if (abs(start) >= x.shape().size() || abs(end) > x.shape().size()) {
	//	throw std::runtime_error("ShapeParams are out of bounds");
	//}
	//xt::xarray<Tr> shape = xt::adapt(x.shape(), { x.shape().size() - start - end }, xt::no_ownership(), x.shape().data() + start);
    int start = params.start;
    int end = params.end;
    if (end == std::numeric_limits<int>::max()) {
        end = 0; 
	}

    if (start < 0) start = x.shape().size() + start; 
    if (end < 0) end = x.shape().size() - end;

    if (start < 0 || start > static_cast<int>(x.shape().size()) ||
        end < 0 || end > static_cast<int>(x.shape().size()) ||
        start + end > static_cast<int>(x.shape().size()))
    {
        throw std::runtime_error("ShapeParams are out of bounds. Given Shapesize: " + std::to_string(x.shape().size()) + " Start: " + std::to_string(start) + " End: " + std::to_string(end));
    }

   
    std::vector<int64_t> sliced_shape(
        x.shape().begin() + start,
        x.shape().end() - end
    );
    r = xt::adapt(sliced_shape);
}

