#include "OnnxNodes.h"
#include "Utils.h"
class GemmHandler : public OperatorHandler {
public:
GemmHandler(const OnnxNode* node) : OperatorHandler(node) {}
bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
std::string GetNodeHandlerString() const override {
	std::string res = "Gemm(";


	if (!node->GetInputNames().empty()) {
		res += join(node->GetInputNames(), ", ");
	}
	if (!node->GetOutputNames().empty()) {
		res += ", " + join(node->GetOutputNames(), ", ");
	}
	if (node->GetAttributes().size() > 0) {
		res += ", " + node->GetParamsString();
	}
	res += "); // " + node->GetName();


	return res;
}
};
REGISTER_OPERATOR_HANDLER(GemmHandler, "Gemm")
