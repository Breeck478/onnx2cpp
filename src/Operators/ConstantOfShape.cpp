#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
using namespace toCpp;
class ConstantOfShapeHandler : public OperatorHandler {
public:
	ConstantOfShapeHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	bool OperatorNeedsInclude() const override {
		return false; // This operator does not need an include
	}
	bool OperatorSpecificTensorTypes() const override {
		return true; // This operator has specific tensor types
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			if (node->GetOutputNames().size() == 1 && node->GetInputNames().size() == 1) {

				if (node->GetAttribute("value").index()) { // index() returns 0 if the attribute is not set
						auto value = std::get<onnx::TensorProto>((node->GetAttribute("value")));
						OnnxConst constant(value);
						
						if (constant.GetDataSize() != 1)
							throw std::runtime_error("ERROR(ConstantOfShapeHandler::GetOpSpecificNodeGenString): ConstantOfShape operator must have exactly one value in the Attribute tensor.");

						auto valString = constant.GetDataAsString(false);
						valString.erase(0, 5); // Remove "value" (name of this tensor) mby use finde to remove everything till "="
						valString = RemoveChars(valString, "{} =;"); // Remove curly braces and equal
						// Output shape is simular to shape given by the input
						stream <<  node->GetOutputNames()[0] << ".resize({" << Join(node->GetOutputs()[0]->Shape(), ",") << "});\n";
						stream << node->GetOutputNames()[0] << " = " << "xt::full_like(" << node->GetOutputNames()[0] << ", " << valString << ");\n";
					}
					else {
						stream <<  node->GetOutputNames()[0] << ".resize({" << Join(node->GetOutputs()[0]->Shape(), ",") << "});\n";
						stream << node->GetOutputNames()[0] << " = " << "xt::full_like(" << node->GetOutputNames()[0] << ", 0);\n";
					}
				
			}
			else {
				std::cerr << "Constant of Shape operator must have exactly one output and input." << std::endl;
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Constant operator: " << e.what() << std::endl;
		}
	}
};
REGISTER_OPERATOR_HANDLER(ConstantOfShapeHandler, "ConstantOfShape")
