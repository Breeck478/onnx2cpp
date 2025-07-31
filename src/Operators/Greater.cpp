#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
class GreaterHandler : public OperatorHandler {
public:
	GreaterHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return true; // This operator has specific generation logic
	}
	void SetTensorTypes() override {
		// Do nothing. outputtypes are always bool independent on what the input type is
	}
};
REGISTER_OPERATOR_HANDLER(GreaterHandler, "Greater")
