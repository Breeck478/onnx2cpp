#include "OnnxNode.h"
#include "Utils.h"
using namespace toCpp;
class GemmHandler : public OperatorHandler {
public:
	GemmHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return false; // This operator has specific generation logic
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {
		stream << "Gemm(";


		if (!node->GetInputNames().empty()) {
			stream << Join(node->GetInputNames(), ", ");
		}
		if (!node->GetOutputNames().empty()) {
			stream << ", " + Join(node->GetOutputNames(), ", ");
		}
		if (node->GetAttributes().size() > 0) {
			stream << ", " + node->GetParamsString();
		}
		stream << "); // " + node->GetName();
	}
};
REGISTER_OPERATOR_HANDLER(GemmHandler, "Gemm")
