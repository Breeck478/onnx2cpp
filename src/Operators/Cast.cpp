#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <algorithm>  

#include <any>

#include <string>
using namespace toCpp;
class CastHandler : public OperatorHandler {
public:
	CastHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	bool OperatorSpecificTensorTypes() const override {
		return true; // The output type is alway given by the output Tensor type
	}
	void PrePrint() override {
		if (node->GetOutputs().size() < 1) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Cast operator has no outputs");
		}

		int32_t realType = node->GetOutputs()[0]->DataType(); // Get Output type
		auto it = node->GetAttribute("to");
		if (!it.has_value()) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Cast operator has no 'to' attribute");
		}
		
		int32_t expectedType = -1;
		try {
			expectedType = std::any_cast<int32_t>(it);
		}
		catch (const std::bad_any_cast& e) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Bad cast for 'to' attribute: " + std::string(e.what()));
		}
		if (realType != expectedType) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Cast operator output type does not match 'to' attribute type");
		}
		
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			int64_t to = std::any_cast<int64_t>(node->GetAttribute("to"));
			std::ostringstream tmpStore;
			tmpStore.swap(stream);
			stream << "#ifdef DCO_ENABLE_EXPLICIT_TYPE_CAST_TO \n";
			stream << "DCO_ENABLE_EXPLICIT_TYPE_CAST_TO(" + GetDataTypeString(to) + ")\n";
			stream << "#endif  \n";
			stream << tmpStore.str();
			node->CreateFunctionCall(stream);
		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Cast operator: " << e.what() << std::endl;
		}
	}
};
REGISTER_OPERATOR_HANDLER(CastHandler, "Cast")
