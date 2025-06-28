#include <xtensor/xarray.hpp>

template<typename T>
void Tanh(const xt::xarray<T>& x, xt::xarray<T>& r){
  r = xt::sin(x);
}




