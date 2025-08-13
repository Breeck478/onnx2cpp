#include "onnx2cpp.h"

#include "Utils.h"
#include <onnx/shape_inference/implementation.h>
#include "onnx/defs/schema.h"
using namespace toCpp;
using namespace std;

void onnx2cpp::ParseInputs(int argc, char* argv[]) {
	cout << "\nCommand-line arguments:\n";
	int count;
	bool fileInSet = false;
	bool fileOutSet = false;
	for (count = 0; count < argc; count++) {
		if (string(argv[count]) == "--fileIn") {
			if (count + 1 < argc) {
				modelFileName = argv[count + 1];
				count++;
				fileInSet = true;
			}
			else {
				cout << "Error: --fileIn option requires a file name argument." << endl;
			}
		}
		else if (string(argv[count]) == "--fileOut") {
			if (count + 1 < argc) {
				outputFileName = argv[count + 1];
				count++;
			}
			else {
				cout << "Error: --fileOut option requires a file name argument." << endl;
			}
		}
		else if (string(argv[count]) == "--static_inputs") {
			if (count + 1 < argc) {
				string inputs = argv[count + 1];
				if (inputs[0] == '[' && inputs[inputs.size() - 1] == ']') {
					inputs = RemoveChars(inputs, "[] "); // Remove quotes and spaces
					staticInputs = split(inputs, ",");
				}
				else {
					cout << "Error: --static_inputs not set proberly" << endl;
				}
				count++;
			}
		}
		else if (string(argv[count]) == "--allStatic") {
			allStatic = true;
		}
	}
	if (!fileInSet || modelFileName.empty()) {
		cout << "Error: No input file provided" << endl;
		exit(1);
	}

	if (!modelFileName.ends_with(".onnx")) {
		modelFileName += ".onnx"; // Add default extension if not present
	}
	if (!fileOutSet || outputFileName.empty()) {
			outputFileName = modelFileName.substr(0, modelFileName.size() - 5);
			outputFileName += ".h"; // Add default extension if not present
	}
	else if (!outputFileName.ends_with(".h")) {
		outputFileName += ".h";
	}
}

string onnx2cpp::MakeCppFile(onnx::ModelProto &model, ostream &stream, vector<string> staticInputs) {
		try
		{
			onnx::shape_inference::DataValueMap aMap;
			onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
			onnx::shape_inference::SymbolTableImpl symbolTable;
			symbolTable.addFromGraph(model.graph());
			onnx::GraphProto graphProto = model.graph();
			OnnxGraph graph(graphProto, true, staticInputs);
			graph.PreProcess();
			ostringstream oss;
			graph.PrintGraph(oss);
			stream << oss.str();
			return graph.Name(); // return graph name for function call in testsuit
		}
		catch (const std::exception& e)
		{
			cout << "Error: " << e.what() << endl;
			return ""; // Return empty string if an error occurs
		}
		return ""; // Return empty string if an error occurs

}

OnnxGraph onnx2cpp::MakeCppFileGraphOut(onnx::ModelProto& model, ostream& stream, vector<string> staticInputs) {
	try
	{
		onnx::shape_inference::DataValueMap aMap;
		onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
		onnx::shape_inference::SymbolTableImpl symbolTable;
		symbolTable.addFromGraph(model.graph());
		onnx::GraphProto graphProto = model.graph();
		OnnxGraph graph(graphProto, true, staticInputs);
		graph.PreProcess();
		ostringstream oss;
		graph.PrintGraph(oss);
		stream << oss.str();
		return graph; // return graph name for function call in testsuit
	}
	catch (const std::exception& e)
	{
		cout << "Error: " << e.what() << endl;
		return OnnxGraph(); // Return empty string if an error occurs
	}
	return OnnxGraph(); // Return empty string if an error occurs

}

string onnx2cpp::MakeCppFile(onnx::ModelProto& model, ostream& stream, bool allStatic) {
	try
	{
		onnx::shape_inference::DataValueMap aMap;
		onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
		onnx::shape_inference::SymbolTableImpl symbolTable;
		symbolTable.addFromGraph(model.graph());
		onnx::GraphProto graphProto = model.graph();
		std::vector<std::string> staticInputs;
		if (allStatic) {
			for (const auto& input : model.graph().input())
				staticInputs.push_back(input.name());
		}
		OnnxGraph graph(graphProto, true, staticInputs);
		graph.PreProcess();
		ostringstream oss;
		graph.PrintGraph(oss);
		stream << oss.str();
		return graph.Name(); // return graph name for function call in testsuit
	}
	catch (const std::exception& e)
	{
		cout << "Error: " << e.what() << endl;
		return ""; // Return empty string if an error occurs
	}
	return ""; // Return empty string if an error occurs

}





