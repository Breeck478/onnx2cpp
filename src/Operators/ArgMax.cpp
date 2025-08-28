#include "OnnxNode.h"
#include "OnnxConst.h"
using namespace toCpp;
class ArgMinHandler : public OperatorHandler {
public:
	ArgMinHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return true; // This operator has specific generation logic
	}
	void SetOpSpecificTensorTypes() override {
		// Do nothing. outputtypes are always int64_t independent on what the input type is
	}
};
REGISTER_OPERATOR_HANDLER(ArgMinHandler, "ArgMin")
