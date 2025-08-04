#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
class EqualHandler : public OperatorHandler {
public:
	EqualHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return true; // This operator has specific generation logic
	}
	void SetOpSpecificTensorTypes() override {
		// Do nothing. outputtypes are always bool. Independent on what the input type is
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(EqualHandler, "Equal")
