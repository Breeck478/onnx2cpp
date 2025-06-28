#include "OnnxNodes.h"
#include "OnnxConsts.h"
#include "Utils.h"

#include <string>
class ConstantHandler : public OperatorHandler {
public:
	ConstantHandler(OnnxNode node) : OperatorHandler(node) {}
	bool OperatorSpecificGeneration() const override {
		return true; // This operator has specific generation logic
	}
	std::string GetVarInitString() const override {

		std::string res = "";
		try {
			if (node.GetOutputs().size() > 0 && node.GetInputs().size() == 0) {
				for (auto att : node.GetAttributes())
				{
					if (att.first == "value" || att.first == "sparse_value") {
						auto value = std::any_cast<onnx::TensorProto>(att.second);
						OnnxConst constant(value);
						constant.SetName(node.GetOutputs()[0]);
						res += constant.GetVarInitString();
					}
					else if (att.first.rfind("value_", 0) == 0) {
						std::cout << att.first << std::endl;
						res += att.first + " = ";
					}
					else {
						std::cerr << "Unknown attribute in Constant operator: " << att.first << std::endl;
					}
				}
			}
			else {
				std::cerr << "Constant operator must have exactly one output and no inputs." << std::endl;
				return "";
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Constant operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(ConstantHandler, "Constant")
