#include "OnnxNode.h" 
#include "Utils.h"
using namespace toCpp;
class ConcatHandler : public OperatorHandler {
public:
	ConcatHandler(const OnnxNode* node) : OperatorHandler(node) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			if (node->GetInputNames().size() > 0) {
				std::string vectorName = "vectorForConcat" + std::to_string(count++);
				stream << "const std::vector<xt::xarray<" << node->GetInputs()[0]->GetDataTypeAsString() << ">> "<< vectorName  << " = {" + join(node->GetInputNames(), ", ") + "}; \n";
				stream << "Concat("<< vectorName <<", " + node->GetOutputNames()[0];
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
private:
	static int count; // Static counter for having unique names for different ConcatHandler instances
};
int ConcatHandler::count = 0; // Initialize static counter
REGISTER_OPERATOR_HANDLER(ConcatHandler, "Concat")
