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
		try {		
			// create map without aaded vars for later use
			std::map<OnnxVar*, OnnxVar*> inToOut; // input var to output var 
			for (int i = 1; i < Graph().GetInputs().size(); i++) {
				inToOut[Graph().GetInputs()[i]] = Graph().GetOutputs()[i - 1];
			}
			// Add Vars from the main(outer) graph to the Loop Graph
			Graph().AddExternVars(node->GetGraph()->GetVars());
			// Mark the inputs and outputs of the Loop Graph as static or non-static 
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
			// Set static ins and outs
			this->graph.SetStaticIOs(inputNames, outputNames);

			// Now check map, wether in and out types do match. If one of them is non static the other one has to be static as well
			// Could check here if the static types would be correct
			for (auto [in, out] : inToOut) {
				if (!in->HasStaticType() || !out->HasStaticType()) {
					out->HasStaticType(false); 
					in->HasStaticType(false); 
				}
			}
			
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
				
				std::vector<std::string> inputNames;
				std::vector<std::string> outputNames;
				for (int i = 0; i < node->GetInputNames().size(); i++) {
					inputNames.push_back(node->GetInputNames()[i] + "_In");
					if (i == 0)
						continue;
					outputNames.push_back(node->GetInputNames()[i] + "_Out");
				}
				res += Graph().PrintGraph();
				for (int i = 1; i < inputNames.size(); i++) {
					res += "xt::xarray<" + Graph().GetInputs()[i]->GetDataTypeAsString() + "> " + inputNames[i] + " = " + node->GetInputNames()[i] + ";\n";
				}
				for (int i = 0; i < outputNames.size(); i++) {
					res += "xt::xarray<" + Graph().GetInputs()[i+1]->GetDataTypeAsString() + "> " + outputNames[i] + ";\n";
				}
				res += "// Loop Graph:\n";
				res += "for (int " + inputNames[0] + " = 0; "+inputNames[0]+" < " + node->GetInputNames()[0] + "[0] && " + inputNames[1] + "[0]; ++" + inputNames[0] + ") {\n";
				res += "\t// Loop body for " + node->GetName() + "\n";
				res += Graph().Name() + "(" + join(inputNames, ", ") + ", " + join(outputNames, ", ") + "); // " + node->GetName() + "\n";
				for (int i = 0; i+1 < inputNames.size() && i < outputNames.size(); i++) {
					res += inputNames[i+1] + " = " + outputNames[i] + ";\n";
				}
				res += "}\n";
				for (int i = 0; i < node->GetOutputNames().size() && i + 1 < outputNames.size(); i++) {
					res += node->GetOutputNames()[i] + " = " + outputNames[i + 1] + ";\n";
				}

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


	const OnnxGraph& Graph() const {
		return graph;
	}
	
	OnnxGraph& Graph() {
		return graph;
	}
private:
	OnnxGraph graph;
};
REGISTER_OPERATOR_HANDLER(LoopHandler, "Loop")
