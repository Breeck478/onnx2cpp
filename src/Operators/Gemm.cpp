#include "OnnxNodes.h"
#include "Utils.h"
class GemmHandler : public OperatorHandler {
public:
	GemmHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	void GetNodeHandlerString(std::ostringstream & stream) const override {
		stream << "Gemm(";


		if (!node->GetInputNames().empty()) {
			stream << join(node->GetInputNames(), ", ");
		}
		if (!node->GetOutputNames().empty()) {
			stream << ", " + join(node->GetOutputNames(), ", ");
		}
		if (node->GetAttributes().size() > 0) {
			stream << ", " + node->GetParamsString();
		}
		stream << "); // " + node->GetName();
	}
};
REGISTER_OPERATOR_HANDLER(GemmHandler, "Gemm")
