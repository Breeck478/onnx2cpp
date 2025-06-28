#include "OnnxNodes.h" 
#include "Utils.h"

class ConcatHandler : public OperatorHandler {
public:
	ConcatHandler(OnnxNode node) : OperatorHandler(node) {}
	bool OperatorSpecificGeneration() const override {
		return true; // This operator has specific generation logic
	}
	std::string GetVarInitString() const override {

		std::string res = "";
		try {
			if (node.GetInputs().size() > 0) {
				res += "Concat(std::tuple(" + join(node.GetInputs(), ", ") + "), " + node.GetOutputs()[0];
				if (node.GetAttributes().size() > 0) {
					res += ", " + node.GetParamsString();
				}
				res += "); // " + node.GetName();
			}
			else {
				std::cerr << "Concat operator must have inputs." << std::endl;
				return "";
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Concat operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(ConcatHandler, "Concat")
