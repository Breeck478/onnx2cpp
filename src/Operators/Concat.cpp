#include "OnnxNode.h" 
#include "Utils.h"

class ConcatHandler : public OperatorHandler {
public:
	ConcatHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	void GetNodeHandlerString(std::ostringstream & stream) const override {

		try {
			if (node->GetInputNames().size() > 0) {
				stream << "std::vector<xt::xarray<T>> myVector = {" + join(node->GetInputNames(), ", ") + "};\n";
				stream << "Concat(myVector, " + node->GetOutputNames()[0];
				if (node->GetAttributes().size() > 0) {
					stream << ", " + node->GetParamsString();
				}
				stream << "); // " + node->GetName();
			}
			else {
				std::cerr << "Concat operator must have inputs." << std::endl;
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Concat operator: " << e.what() << std::endl;
		}
	}
};
REGISTER_OPERATOR_HANDLER(ConcatHandler, "Concat")
