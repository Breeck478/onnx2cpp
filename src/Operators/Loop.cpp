#include "OnnxNode.h"
#include "OnnxGraph.h"
#include "Utils.h"

#include <any>

#include <string>

#include <vector>
#include <map>
using namespace toCpp;

class LoopHandler : public OperatorHandler {
public:
	LoopHandler(const OnnxNode* node) : OperatorHandler(node) {
		if (std::holds_alternative<onnx::GraphProto>(node->GetAttribute("body")))
			this->graph = OnnxGraph(std::get<onnx::GraphProto>(node->GetAttribute("body")), false);
	}
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
		return true; 
	}
	virtual void SetOpSpecificTensorTypes() {
			

		// Add all tensors from the main(outer) graph to the Loop Graph
		Graph().AddExternVars(node->GetGraph()->GetVars());
		Graph().AddExternConsts(node->GetGraph()->GetConsts());
		// Fill Map for graph in- and outputs
		for (int64_t i = 0; i < Graph().GetOutputs().size(); i++) { // outputs size might be bigger then inputs because of scans
			if (i +1 < Graph().GetInputs().size())
				OutToIn().push_back(std::pair(Graph().GetOutputs()[i], Graph().GetInputs()[i + 1])); 
			else 
				OutToIn().push_back(std::pair(Graph().GetOutputs()[i] , nullptr)); // Scan output
			
		}
		// Fill map for node inputs to graph inputs
		for (int64_t i = 0; i < node->GetInputs().size(); i++) { // start at one because 
			if (i < Graph().GetInputs().size())
				InToIn().push_back(std::pair(node->GetInputs()[i], Graph().GetInputs()[i]));
		}
		// Fill map for graph outputs to node outputs
		for (int64_t i = 1; i < Graph().GetOutputs().size(); i++) { // Skip first one because it is the condition which is not given back as result of node
			if (i-1 < node->GetOutputs().size())
				OutToOut().push_back(std::pair(Graph().GetOutputs()[i], node->GetOutputs()[i-1]));
		}

		// Mark the inputs and outputs of the Loop Graph as static or dynamic
		std::vector<std::string> inputNames = Graph().GetInputNames();
		for (int64_t i = inputNames.size() - 1; i >= 1; i--) { // start at 1 to ignore the iterater input
			if ((i < node->GetInputs().size()) && (!node->GetInputs()[i]->HasStaticType())) {
				inputNames.erase(inputNames.begin() + i);
			}
		}
		Graph().SetStaticIOs(inputNames);
		// Now check map, wether in and out types do match. If one of them is dynamic the other one has to be dynamic as well
		for (auto& [in, out] : OutToIn()) {
			if (!in->HasStaticType() || (out != nullptr && !out->HasStaticType())) {
				if (out != nullptr)
					out->HasStaticType(false);
				in->HasStaticType(false);
			}
		}
		for (auto& [graphOut, nodeOut] : OutToOut()) {
			if (!graphOut->HasStaticType()) {
				nodeOut->HasStaticType(false);
			}
		}
	}
	void PrePrint() override {	

	}
	void GetOpSpecificNodeGenString(std::ostringstream & stream) const override {

		try {
			if (node->GetInputs().size() >= 2 && node->GetOutputs().size() >= 1 && node->GetAttributes().size() == 1) {
				
				std::vector<std::string> inputNames;
				std::vector<std::string> outputNames;
				int amountScans = Graph().GetOutputNames().size() - (Graph().GetInputNames().size() - 1);
				if (amountScans < 0) {
					amountScans = 0; // No scans needed
				}  
				for (int64_t i = 0; i < Graph().GetInputNames().size(); i++) {
					inputNames.push_back(Graph().GetInputNames()[i] + "_in");

				}

				for (int64_t i = 0; i < Graph().GetOutputNames().size(); i++) {
					outputNames.push_back(Graph().GetOutputNames()[i] + "_out");
				}
				Graph().PrintGraph(stream);
				for (int64_t i = 1; i < inputNames.size(); i++) {
					stream << "xt::xarray<" + Graph().GetInputs()[i]->GetDataTypeAsString() + "> " + inputNames[i] + " = " + node->GetInputNames()[i] + ";\n";
				}
				for (int64_t i = 0; i < outputNames.size(); i++) {
					stream << "xt::xarray<" + Graph().GetOutputs()[i]->GetDataTypeAsString() + "> " + outputNames[i] + ";\n";
				}
				size_t nodeOutputsSize = node->GetOutputNames().size() - 1;
				for (int64_t i = 0; i < amountScans; i++) {
					stream << node->GetOutputNames()[nodeOutputsSize-i] + ".resize({" << Join(node->GetOutputs()[nodeOutputsSize - i]->Shape(), ", ") << "});\n";
				}
				
				stream << "// Loop Graph:\n";
				stream << "for (size_t " + inputNames[0] + " = 0; "+inputNames[0]+" < " + node->GetInputNames()[0] + "[0] && " + inputNames[1] + "[0]; ++" + inputNames[0] + ") {\n";
				stream << "\t// Loop body for " + node->GetName() + "\n";
				stream << Graph().Name() + "(" + Join(inputNames, ", ") + ", " + Join(outputNames, ", ") + "); // " + node->GetName() + "\n";
				for (int64_t i = 0; i+1 < inputNames.size() && i < outputNames.size(); i++) {
					stream << inputNames[i+1] + " = " + outputNames[i] + ";\n";
				}
				for (int64_t i = 0; i < amountScans; i++) {
					stream << "xt::view(" << node->GetOutputNames()[nodeOutputsSize - i] << ", " << inputNames[0] << ", xt::all()) = xt::view(" << outputNames[outputNames.size() - 1 - i] + ", xt::all()); \n";
				}
				stream << "}\n";
				for (int64_t i = 0; i < node->GetOutputNames().size() - amountScans && i + 1 < outputNames.size(); i++) {
					stream << node->GetOutputNames()[i] + " = " + outputNames[i + 1] + ";\n";
				}

			}
			else {
				std::cerr << "Loop operator must have at least two inputs, one output and exactly one Graph attribute" << std::endl;
			}

		}
		catch (const std::exception& e) {
			std::cerr << "Error generating Loop operator: " << e.what() << std::endl;
		}
	}


	const OnnxGraph& Graph() const {
		return graph;
	}
	
	OnnxGraph& Graph() {
		return graph;
	}
	std::vector<std::pair<OnnxTensor*, OnnxTensor*>>& InToIn() {
		return inToIn;
	}
	std::vector<std::pair<OnnxTensor*, OnnxTensor*>>& OutToOut() {
		return outToOut;
	}
	std::vector<std::pair<OnnxVar*, OnnxVar*>>& OutToIn() {
		return outToIn;
	}
private:
	OnnxGraph graph;
	std::vector<std::pair<OnnxTensor*, OnnxTensor*>> inToIn; // Node in to Graph in 
	std::vector<std::pair<OnnxTensor*, OnnxTensor*>> outToOut; // Graph out to Node out
	std::vector<std::pair<OnnxVar*, OnnxVar*>> outToIn; // Graph in to Greaph out
};
REGISTER_OPERATOR_HANDLER(LoopHandler, "Loop")
