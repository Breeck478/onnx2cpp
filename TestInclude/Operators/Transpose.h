#include <xtensor.hpp>
#include <vector>

struct TransposeParams {
	const std::vector<int> perm;
};

template<typename T>
void Transpose(const xt::xarray<T>& data, xt::xarray<T>& transposed, const TransposeParams& params = TransposeParams()) {
	
	if (params.perm.empty()) {
		transposed = xt::transpose(data);
		return;
	}
	if (params.perm.size() != data.shape().size())
		throw std::runtime_error("Tranpose: Permutation has to be the same length as rank of input" );

	transposed = xt::transpose(data, params.perm);
}



