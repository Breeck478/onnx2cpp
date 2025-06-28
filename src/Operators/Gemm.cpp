#include "OnnxNodes.h"
#include "Utils.h"
class GemmHandler : public OperatorHandler {
public:
GemmHandler(OnnxNode node) : OperatorHandler(node) {}
bool OperatorSpecificGeneration() const override {
		return true; // This operator has specific generation logic
	}
std::string GetVarInitString() const override {
	std::string res = "Gemm(";


	if (!node.GetInputs().empty()) {
		res += join(node.GetInputs(), ", ");
	}
	if (!node.GetOutputs().empty()) {
		res += ", " + join(node.GetOutputs(), ", ");
	}
	if (node.GetAttributes().size() > 0) {
		res += ", " + node.GetParamsString();
	}
	res += "); // " + node.GetName();


	return res;
}



//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(GemmHandler, "Gemm")
