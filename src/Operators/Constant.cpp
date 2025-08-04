#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
class ConstantHandler : public OperatorHandler {
public:
	ConstantHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	bool OperatorSpecificVarGeneration() const override {
		return false; // This operator does not generate specific variables
	}
	bool OperatorNeedsInclude() const override {
		return false; // This operator does not need an include
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			if (node->GetOutputNames().size() > 0 && node->GetInputNames().size() == 0) {
				for (auto att : node->GetAttributes())
				{
					if (att.first == "value" || att.first == "sparse_value") {
						auto value = std::any_cast<onnx::TensorProto>(att.second);
						OnnxConst constant(value);
						constant.Name(node->GetOutputNames()[0]);
						stream << constant.GetConstantString(false);
					}
					else if (att.first.rfind("value_", 0) == 0) {
						std::cout << att.first << std::endl;
						stream << att.first + " = ";
					}
					else {
						std::cerr << "Unknown attribute in Constant operator: " << att.first << std::endl;
					}
				}
			}
			else {
				std::cerr << "Constant operator must have exactly one output and no inputs." << std::endl;
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Constant operator: " << e.what() << std::endl;
		}
	}
};
REGISTER_OPERATOR_HANDLER(ConstantHandler, "Constant")
