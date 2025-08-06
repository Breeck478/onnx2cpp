#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
using namespace toCpp;
class WhereHandler : public OperatorHandler {
public:
	WhereHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return false; // This operator has specific generation logic
	}
	void SetOpSpecificTensorTypes() override {
		// Do nothing. outputtypes are always boll independent on what the input type is
	}
};
REGISTER_OPERATOR_HANDLER(WhereHandler, "Where")
