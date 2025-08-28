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

		int64_t realType = node->GetOutputs()[0]->DataType(); // Get Output type
		auto it = node->GetAttribute("to");
		if (!std::holds_alternative<int64_t>(it)) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Cast operator has no 'to' attribute");
		}
		int64_t	expectedType = std::get<int64_t>(it);
		if (realType != expectedType) {
			throw std::runtime_error("ERROR(CastHandler::PrePrint): Cast operator output type does not match 'to' attribute type");
		}
		
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			// write dco-makro at the start of the given stream
			int64_t to = std::get<int64_t>(node->GetAttribute("to"));
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
