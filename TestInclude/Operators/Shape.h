#include <xtensor.hpp>
#include <limits>
#include <optional>
struct ShapeParams
{
	const std::optional<int> end;
	const int start = 0;
};
bool _interval(const int& n,const ShapeParams& params, int& absStart, int& absEnd) {
    if (params.start == 0) {
        if (!params.end.has_value()) {
            return false;
        }
        if (params.end.has_value() && params.end.value() < 0) {
            absStart = 0;
			absEnd = n + params.end.value();
            return true;
        }
        absStart = 0;
        absEnd = n;
        return true;
    }
    if (!params.end.has_value()) {
        absStart = params.start;
        absEnd = n;
        return true;
    }
    if (params.end < 0) {
        absStart = 0;
        absEnd = n + params.end.value();
        return true;
    }
    absStart = params.start;
    absEnd = n + params.end.value();
    return true;
        
}
template <typename T>
xt::xarray<T> appendValue(const xt::xarray<T>& arr, T val)
{
    xt::xarray<T> new_arr = xt::empty<T>({ arr.size() + 1 });
    std::copy(arr.begin(), arr.end(), new_arr.begin());
    new_arr(arr.size()) = val;
    return new_arr;
}
template <typename T>
void Shape(const xt::xarray<T>& data, xt::xarray<int64_t>& shape, const ShapeParams& params = ShapeParams())
{
    xt::xarray<int64_t> dataShape;
    for (size_t i = 0; i < data.shape().size(); i++) {
        if (i == 0) {
            dataShape.data()[i] = data.shape()[i];
        }
        else {
            int64_t entry = data.shape()[i];
            dataShape = appendValue(dataShape, entry);
        }

    }
    int absStart = 0;
    int absEnd = 0;
    if (!_interval(data.shape().size(), params, absStart, absEnd)) {
        shape = dataShape;
    }
    else {
        shape = xt::view(dataShape, xt::range(_, absStart), xt::range(absEnd, _));
    }
}

