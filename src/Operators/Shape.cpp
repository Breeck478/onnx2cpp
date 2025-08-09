#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
using namespace toCpp;
class ShapeHandler : public OperatorHandler {
public:
	ShapeHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificTensorTypes() const override {
		return true; // This operator has specific generation logic
	}
	void SetOpSpecificTensorTypes() override {
		// Do nothing. outputtypes are always int64_t independent on what the input type is
	}
};
REGISTER_OPERATOR_HANDLER(ShapeHandler, "Shape")
