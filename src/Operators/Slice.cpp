#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
using namespace toCpp;
class SliceHandler : public OperatorHandler {
public:
	SliceHandler(const OnnxNode* node) : OperatorHandler(node) {}
	void PrePrint() override{
		if (node->GetAttributes().size() > 0) {
			throw std::runtime_error("Slice: Opset version must be 10 or higher. Versions below are not supported");
		}
	}
};
REGISTER_OPERATOR_HANDLER(SliceHandler, "Slice")
