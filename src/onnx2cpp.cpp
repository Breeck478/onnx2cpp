#include "onnx2cpp.h"

#include "Utils.h"
#include <onnx/shape_inference/implementation.h>
#include "onnx/defs/schema.h"
using namespace toCpp;
int64_t OnnxTensor::batchSize = 1; // Default batch size, can be changed by user input
using namespace std;

void onnx2cpp::ParseInputs(int argc, char* argv[]) {
	cout << "\nCommand-line arguments:\n";
	int count;
	for (count = 0; count < argc; count++) {
		if (string(argv[count]) == "--fileIn") {
			if (count + 1 < argc) {
				modelFileName = argv[count + 1];
				count++;
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
		else if (string(argv[count]) == "--dynamic_dim") {
			if (count + 1 < argc) {
				batchSize = stoi(argv[count + 1]);
				count++;
			}
			else {
				cout << "Error: --batch_size option requires a batch size argument." << endl;
			}
		}
		else if (string(argv[count]) == "--static_inputs") {
			if (count + 1 < argc) {
				string inputs = argv[count + 1];
				if (inputs[0] == '[' && inputs[inputs.size() - 1] == ']') {
					inputs = remove_chars(inputs, "[] "); // Remove quotes if they are present
					staticInputs = split(inputs, ",");
				}
				else {
					cout << "Error: --static_inputs not set proberly" << endl;
				}
			}
		}
		else if (string(argv[count]) == "--static_outputs") {
			if (count + 1 < argc) {
				string outputs = argv[count + 1];
				if (outputs[0] == '"' && outputs[-1] == '"') {
					outputs = remove_chars(outputs, "\" "); // Remove quotes if they are present
					staticOutputs = split(outputs, ",");
				}
				else {
					cout << "Error: --static_outputs not set proberly" << endl;
				}
			}
		}
	}
}

string onnx2cpp::MakeCppFile(onnx::ModelProto &model, ostream &stream, int batchSize, vector<string> staticInputs, vector<string> staticOutputs) {
		try
		{
			onnx::shape_inference::DataValueMap aMap;
			onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
			onnx::shape_inference::SymbolTableImpl symbolTable;
			symbolTable.addFromGraph(model.graph());
			onnx::GraphProto graphProto = model.graph();
			OnnxGraph graph(graphProto, true, staticInputs, staticOutputs);
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

OnnxGraph onnx2cpp::MakeCppFileGraphOut(onnx::ModelProto& model, ostream& stream, bool allStatic) {
	try
	{
		onnx::shape_inference::DataValueMap aMap;
		onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
		onnx::shape_inference::SymbolTableImpl symbolTable;
		symbolTable.addFromGraph(model.graph());
		onnx::GraphProto graphProto = model.graph();
		std::vector<std::string> staticInputs;
		std::vector<std::string> staticOutputs;
		if (allStatic) {
			for (const auto& input : model.graph().input()) 
				staticInputs.push_back(input.name());
			for (const auto& output : model.graph().output()) 
				staticOutputs.push_back(output.name());
		}
		OnnxGraph graph(graphProto, true, staticInputs, staticOutputs);
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
		std::vector<std::string> staticOutputs;
		if (allStatic) {
			for (const auto& input : model.graph().input())
				staticInputs.push_back(input.name());
			for (const auto& output : model.graph().output())
				staticOutputs.push_back(output.name());
		}
		OnnxGraph graph(graphProto, true, staticInputs, staticOutputs);
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





