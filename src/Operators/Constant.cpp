#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"
#include <string>
using namespace toCpp;
class ConstantHandler : public OperatorHandler {
public:
	ConstantHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	bool OperatorNeedsInclude() const override {
		return false; // This operator does not need an include
	}
	void GetOpSpecificNodeGenString(std::ostringstream& stream) const override {

		try {
			if (node->GetOutputNames().size() > 0 && node->GetInputNames().size() == 0) {
				for (auto& att : node->GetAttributes())
				{
					if (att.first == "value") {
						auto value = std::any_cast<onnx::TensorProto>(att.second);
						OnnxConst constant(value);
						constant.Name(node->GetOutputNames()[0]);
						stream << constant.GetConstantString(false);
					}
					else if (att.first.starts_with("value_")) { //starts with "value_"
						if (node->GetOutputs().size() == 1) {
							OnnxTensor* outputVar = node->GetOutputs()[0];
							stream << outputVar->Name() << " = { ";
							if (att.first == "value_ints") {
								auto value = std::any_cast<std::vector<int64_t>>(att.second);
								 stream << Join(value, ", ");
							}
							else if (att.first == "value_int") {
								auto value = std::any_cast<int64_t>(att.second);
								stream << value;
							}
							else if (att.first == "value_floats") {
								auto value = std::any_cast<std::vector<float>>(att.second);
								stream  << Join(value, ", ");
							}
							else if (att.first == "value_float") {
								auto value = std::any_cast<float>(att.second);
								stream << value;
							}
							else if (att.first == "value_strings") {
								auto value = std::any_cast<std::vector<std::string>>(att.second);
								stream << Join(value, ", ");
							}
							else if (att.first == "value_string") {
								auto value = std::any_cast<std::string>(att.second);
								stream << value;
							}
							else {
								std::cerr << "Unknown value type in Constant operator: " << att.first << std::endl;
								continue;
							}
							stream << outputVar->Name() << " = };\n";
							stream << outputVar->Name() << " = " << outputVar->Name() << ".reshape({" << Join(outputVar->Shape(), ", ") + "});\n";
						}
						else {
							std::cerr << "Constant operator must have exact one output." << std::endl;
						}
					}
					else if (att.first == "sparse_value")
					{
						std::cerr << "Sparse values are not supported in Constant operator: " << node->GetName() << std::endl;
						continue;
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
