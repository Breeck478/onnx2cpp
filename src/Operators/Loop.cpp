#include "OnnxNodes.h"
#include "OnnxGraph.h"
#include "Utils.h"

#include <any>

#include <string>

#include <vector>
#include <map>

//class LoopGraph : public OnnxGraph {
//public:
//	LoopGraph(onnx::GraphProto& graph, bool isInitial = false, std::vector<std::string> staticInputs = std::vector<std::string>(), std::vector<std::string> staticOutputs = std::vector<std::string>()) : OnnxGraph(graph, isInitial, staticInputs, staticOutputs) {
//		// LoopGraph specific initialization if needed
//	}
//	std::string PrintSpecificGraph(const GraphPosition position) override {
//		std::string res = "";
//		if (position == GraphPosition::Begin) {
//			
//		}
//		else if (position == GraphPosition::End) {
//			
//		}
//		else {
//			res += "// Unknown position in LoopGraph\n";
//		}
//		return res;
//	}
//};


class LoopHandler : public OperatorHandler {
public:
	LoopHandler(const OnnxNode* node) : OperatorHandler(node), graph(std::any_cast<onnx::GraphProto>(node->GetAttribute("body")), false) {}
	bool OperatorSpecificNodeGeneration() const override {
		return true; // This operator has specific generation logic
	}
	bool OperatorNeedsInclude() const override {
		return false; // This operator does not need an include
	}
	bool OperatorSpecificPreProcess() const override {
		return true; 
	}
	bool OperatorSpecificTensorTypes() const override { 
		return false; 
	}
	virtual void SetTensorTypes() {
		node->GetOutputs()[0]->HasStaticType(true); // output 0 is the condition and is always bool
	}
	void PreProcess() override {
		// This operator does not need specific preprocessing
		// but we can add any specific logic if needed in the future
		try {
			onnx::GraphProto graphProto = std::any_cast<onnx::GraphProto>(node->GetAttribute("body"));
			this->graph = OnnxGraph(graphProto, false);
			std::vector<std::string> inputNames = Graph().GetInputNames();
			std::vector<std::string> outputNames = Graph().GetOutputNames();
			//outputNames.erase(outputNames.begin()); // always erase first output arg because it is the output condition which is alway a bool
			//inputNames.erase(inputNames.begin());
			//inputNames.erase(inputNames.begin() + 1);
			for (int i = inputNames.size() - 1; i >= 0; i--) {
				if ((i < (node->GetInputs().size())) && !(node->GetInputs()[i]->HasStaticType())) {
					inputNames.erase(inputNames.begin() + i);
					i--;
				}
			}
			// Ignore first Output value because it is the condition, which is not an output of the node but from the graph itself to set it for the next iterration
			for (int i = outputNames.size() - 1; i > 0; i--) {
				if (((i - 1) < (node->GetOutputs().size())) && !(node->GetOutputs()[i - 1]->HasStaticType())) {
					outputNames.erase(outputNames.begin() + i);
					i--;
				}
			}
			this->graph.SetStaticIOs(inputNames, outputNames);
			
		}
		catch (const std::bad_any_cast& e) {
			std::cerr << "Error casting attribute 'body' to GraphProto for loop-Operator: " << e.what() << std::endl;
			throw; // Re-throw the exception to handle it in the main processing flow
		}
	}
	std::string GetNodeHandlerString() const override {

		std::string res = "";
		try {
			if (node->GetInputs().size() >= 2 && node->GetOutputs().size() >= 1 && node->GetAttributes().size() == 1) {
				
				
				std::vector<std::string> inputNames = Graph().GetInputNames();
				std::vector<std::string> outputNames = Graph().GetOutputNames();
				for (int i = inputNames.size() - 1; i >= 0; i--) {
					if ((i < (node->GetInputs().size())) && !(node->GetInputs()[i]->HasStaticType())) {
						inputNames.erase(inputNames.begin() + i);
						i--;
					}
				}
				// Ignore first Output value because it is the condition, which is not an output of the node but from the graph itself to set it for the next iterration
				for (int i = outputNames.size() - 1; i > 0; i--) {
					if (((i - 1) < (node->GetOutputs().size())) && !(node->GetOutputs()[i - 1]->HasStaticType())) {
						outputNames.erase(outputNames.begin() + i);
						i--;
					}
				}
				OnnxGraph graph = Graph();
				graph.SetStaticIOs(inputNames, outputNames);
				res += "// Node vars in;\n";
				for (auto name : node->GetInputNames())
					res += "// " + name + ";\n";
				res += "\n// Node vars out;\n";
				for (auto name : node->GetOutputNames())
					res += "// " + name + ";\n";
				res += "\n// Graph vars in;\n";
				for (auto name : inputNames)
					res += "// " + name + ";\n";
				res += "\n// Graph vars out;\n";
				for (auto name : outputNames)
					res += "// " + name + ";\n";
				std::string condOutputName = node->GetInputNames()[1]+"Output";
				res += graph.PrintGraph();
				res += "xt::xarray<" + node->GetInputs()[1]->GetDataTypeAsString() + "> " + condOutputName + " = " + node->GetInputNames()[1] + ";\n";
				res += "// Loop Graph:\n";
				res += "for (int loopCounter = 0; loopCounter < " + node->GetInputNames()[0] + "[0] && " + node->GetInputNames()[1] + "[0]; ++loopCounter) {\n";
				res += "\t// Loop body for " + node->GetName() + "\n";
				res += graph.Name() + "(" + join(node->GetInputNames(), ", ") + ", " + condOutputName + ", " + join(node->GetOutputNames(), ", ") + "); // " + node->GetName() + "\n";
				res += node->GetInputNames()[1] + " = " + condOutputName + ";\n";
				for (int i = 2; i < node->GetInputs().size() && i - 2 < node->GetOutputs().size(); i++) {
					res += node->GetInputNames()[i] + " = " + node->GetOutputNames()[i - 2] + ";\n";
				}
				res += "}\n";
			}
			else {
				std::cerr << "Constant operator must have exactly one output and no inputs." << std::endl;
				return "";
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Constant operator: " << e.what() << std::endl;
			return "";
		}
		return res;
	}


	OnnxGraph Graph() const {
		return graph;
	}
private:
	OnnxGraph graph;
};
REGISTER_OPERATOR_HANDLER(LoopHandler, "Loop")
