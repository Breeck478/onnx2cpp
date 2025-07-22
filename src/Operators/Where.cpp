#include "OnnxNodes.h"
#include "OnnxConsts.h"
#include "Utils.h"

#include <string>
class WhereHandler : public OperatorHandler {
public:
	WhereHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return false; // This operator has specific generation logic
	}
	void SetTensorTypes() override {
		// Do nothing. outputtypes are always boll independent on what the input type is
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(WhereHandler, "Where")
