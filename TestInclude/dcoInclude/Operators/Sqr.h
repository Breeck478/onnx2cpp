#include <xtensor/xarray.hpp>

template<typename T>
void Sqr(const xt::xarray<T>& x, xt::xarray<T>& r){
  r = x * x;
}




