#include "OnnxNodes.h"
#include "OnnxConsts.h"
#include "Utils.h"

#include <algorithm>  

#include <any>

#include <string>
class CastHandler : public OperatorHandler {
public:
	CastHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	std::string GetNodeHandlerString() const override {

		std::string res = "";
		try {
			int64_t to = std::any_cast<int64_t>(node->GetAttribute("to"));
			res += "DCO_ENABLE_EXPLICIT_TYPE_CAST_TO(" + GetDataTypeString(to) + ")\n";
			res += node->CreateFunctionCall();
		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Cast operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(CastHandler, "Cast")
