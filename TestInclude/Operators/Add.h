#include <xtensor.hpp>


//a,b and c do not have to be the same type because it could be an dco type
template <typename Tx, typename Ty, typename Tr>
void Add(const xt::xarray<Tx> &a, const xt::xarray<Ty>& b, xt::xarray<Tr> &c)
{
	c = a + b;
}

