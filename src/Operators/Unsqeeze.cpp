#include "OnnxNode.h"
#include "OnnxConst.h"
#include "Utils.h"

#include <string>
using namespace toCpp;
class UnsqeezeHandler : public OperatorHandler {
public:
	UnsqeezeHandler(const OnnxNode* node) : OperatorHandler(node) {}
	void PreProcess() override{
		if (node->GetAttributes().size() > 0) {
			throw std::runtime_error("Unsqeeze: Opset version must be 13 or higher. Versions below are not supported");
		}
	}
};
REGISTER_OPERATOR_HANDLER(UnsqeezeHandler, "Unsqeeze")
