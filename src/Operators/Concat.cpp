#include "OnnxNodes.h" 
#include "Utils.h"

class ConcatHandler : public OperatorHandler {
public:
	ConcatHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	std::string GetNodeHandlerString() const override {

		std::string res = "";
		try {
			if (node->GetInputNames().size() > 0) {
				res += "std::vector<xt::xarray<T>> myVector = {" + join(node->GetInputNames(), ", ") + "};\n";
				res += "Concat(myVector, " + node->GetOutputNames()[0];
				if (node->GetAttributes().size() > 0) {
					res += ", " + node->GetParamsString();
				}
				res += "); // " + node->GetName();
			}
			else {
				std::cerr << "Concat operator must have inputs." << std::endl;
				return "";
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Concat operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}



	//GemmHandler(OnnxNode node) : OperatorHandler(node) {}
};
REGISTER_OPERATOR_HANDLER(ConcatHandler, "Concat")
