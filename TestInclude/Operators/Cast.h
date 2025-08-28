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
	output = xt::cast<Tr>(input);

}
