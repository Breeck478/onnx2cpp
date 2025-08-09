#include <xtensor.hpp>

template<typename T>
void Sqrt(const xt::xarray<T>& x, xt::xarray<T>& y){
  y = xt::sqrt(x);
}




