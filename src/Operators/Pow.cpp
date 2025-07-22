#include "OnnxNodes.h" 
#include "OnnxTensor.h" 
#include "Utils.h"

class PowHandler : public OperatorHandler {
public:
	PowHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return false; // This operator has specific generation logic
	}
	std::string GetNodeHandlerString() const override {

		std::string res = "";
		try {
			auto& nodeInputs = node->GetInputs();
			if (nodeInputs.size() == 2) {
				auto& shape = nodeInputs[1]->Shape();
				res += node->GetOutputNames()[0] + " = ";
				/*int scalar = node.GetInputs()[1]->GetTypeProto().tensor_type().dim();
				for (int i = 0; i < scalar; ++i) {
					res += nodeInputs[0]->GetName();
					if (i < scalar - 1) {
						res += " * ";
					}
				}*/
			}
		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Pow operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(PowHandler, "Pow")
