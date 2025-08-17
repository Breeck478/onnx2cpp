#include <xtensor.hpp>
#include <concepts>
template<typename T>
concept Int32OrInt64 = std::same_as<T, int32_t> || std::same_as<T, int64_t>;
template <typename T1, typename T2>
bool ArrayContains(const xt::xarray<T1> &arr, T2 value) {
    return xt::any(xt::equal(arr, value));
}

struct SliceParams
{
	const xt::xarray<int> axes = {};
    const xt::xarray<int> ends = {};
    const xt::xarray<int> starts = {};
};


// multiple overloads for Slice function because of optional paraemeters
// steps relies on axex. Tehrefor is is always clear which function to call
template <typename T, Int32OrInt64 Tind>
void Slice(const xt::xarray<T> &data, const xt::xarray<Tind> &starts, const xt::xarray<Tind>& ends, const xt::xarray<Tind>& axes, const xt::xarray<Tind>& steps, xt::xarray<T> & output)
{
    xt::xstrided_slice_vector sv({});
    int counter = 0;
    xt::xarray<Tind> copiedAxes = axes;
    for (size_t i = 0; i < copiedAxes.size(); i++) {
        if (copiedAxes(i) < 0) {
            copiedAxes(i) += data.shape().size(); // convert negative axes to positive
        }
    }
    for (size_t i = 0; i < data.shape().size(); ++i)
    {
        if (ArrayContains(copiedAxes, i)) { // if i is in axes 

            int start = starts(counter);
            int end = ends(counter);
            int step = (counter < static_cast<int>(steps.size())) ? steps(counter) : 1;
            counter++;
            if (start < 0) start += data.shape(i);
            if (end < 0) end += data.shape(i);

            sv.push_back(xt::range(start, end, step));
        }
        else {
            sv.push_back(xt::all());
        }
    }

    output =  xt::strided_view(data, sv);
}

template <typename T, Int32OrInt64 Tind>
void Slice(const xt::xarray<T>& data, const xt::xarray<Tind>& starts, const xt::xarray<Tind>& ends, const xt::xarray<Tind>& axes, xt::xarray<T>& output)
{
    xt::xstrided_slice_vector sv({});
    int counter = 0;
    xt::xarray<Tind> copiedAxes = axes;
    for (size_t i = 0; i < copiedAxes.size(); i++) {
        if (copiedAxes(i) < 0) {
            copiedAxes(i) += data.shape().size(); // convert negative axes to positive
        }
    }

    for (size_t i = 0; i < data.shape().size(); ++i)
    {
        if (ArrayContains(copiedAxes, i)) { // if i is in axes 

            int start = starts(counter);
            int end = ends(counter);
            counter++;
            if (start < 0) start += data.shape(i);
            if (end < 0) end += data.shape(i);

            sv.push_back(xt::range(start, end));
        }
        else {
            sv.push_back(xt::all());
        }
    }

    output = xt::strided_view(data, sv);
}
template <typename T, Int32OrInt64 Tind>
void Slice(const xt::xarray<T>& data, const xt::xarray<Tind>& starts, const xt::xarray<Tind>& ends, xt::xarray<T>& output)
{
    xt::xstrided_slice_vector sv({});
    int counter = 0;
    for (size_t i = 0; i < data.shape().size(); ++i)
    {


        int start = starts(counter);
        int end = ends(counter);
        counter++;
        if (start < 0) start += data.shape(i);
        if (end < 0) end += data.shape(i);

        sv.push_back(xt::range(start, end));

    }

    output = xt::strided_view(data, sv);
}


template <typename T, Int32OrInt64 Tind>
void Slice(const xt::xarray<T>& data, xt::xarray<T>& output, const SliceParams& params = SliceParams()) {
    Slice(data, params.starts, params.ends, params.axes, output);
}