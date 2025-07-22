#include "OnnxNodes.h"
#include "OnnxConsts.h"
#include "Utils.h"

#include <string>
class ConstantOfShapeHandler : public OperatorHandler {
public:
	ConstantOfShapeHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	std::string GetNodeHandlerString() const override {

		std::string res = "";
		try {
			if (node->GetOutputNames().size() == 1 && node->GetInputNames().size() == 1) {
				for (auto att : node->GetAttributes())
				{
					if (att.first == "value") {
						auto value = std::any_cast<onnx::TensorProto>(att.second);
						OnnxConst constant(value);
						constant.Name(node->GetOutputNames()[0]);
						res += constant.GetConstantString();
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
				std::cerr << "Constant of Shape operator must have exactly one output and input." << std::endl;
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
REGISTER_OPERATOR_HANDLER(ConstantOfShapeHandler, "ConstantOfShape")
