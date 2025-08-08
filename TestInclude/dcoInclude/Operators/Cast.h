#include <cmath>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
//#include <xtensor/xadapt.hpp>
//#include <xtensor/xstrides.hpp>
//#include <xtensor/xeval.hpp>
struct CastParams
{
  const std::string round_mode = "up";
  const int saturate = 1;
  const int to = 0;
};
template <typename Tx, typename Tr>
void Cast(const xt::xarray<Tx>& x, xt::xarray<Tr>& r, const CastParams& params = CastParams())
{
 /*   std::string round_mode = params.round_mode;
    int saturate = params.saturate;
    int to = params.to;*/
	r = xt::cast<Tr>(x);

}
