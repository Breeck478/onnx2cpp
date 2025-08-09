#include <xtensor.hpp>
#include <optional>
struct CastParams
{
  const std::string round_mode = "up";
  const int saturate = 1;
  const std::optional<int> to = 0;
};
template <typename Tx, typename Tr>
void Cast(const xt::xarray<Tx>& input, xt::xarray<Tr>& output, const CastParams& params = CastParams())
{
	// Check if params.to is set because it is required by onnx and check if Tr matches data type defined by params.to
	/*if (params.to.has_value() && typeid(Tr) != typeid(params.to.value())) {
		throw std::runtime_error("Type mismatch: cannot cast to the specified type.");
	}*/
	output = xt::cast<Tr>(input);

}
