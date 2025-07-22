// onnx2cpp.cpp: Definiert den Einstiegspunkt für die Anwendung.
//
#pragma once
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "onnx2cpp.h"
//#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include <onnx/common/file_utils.h>
#include <onnx/defs/printer.h>
#include <onnx/shape_inference/implementation.h>
#include <vector>
#include <string>

#include <fstream>
#include "OnnxGraph.h"
#include "OnnxConsts.h"
#include "OnnxNodes.h"
#include "Utils.h"
#include <cstdlib>
#include <iomanip>
#include <Eigen/Dense>

#include <xtensor/xarray.hpp>

int64_t OnnxTensor::batchSize = 1; // Default batch size, can be changed by user input
using namespace std;

static void PrintGraphInfo(onnx::GraphProto graph) {
	cout << "Graph: " << onnx::ProtoToString(&graph) << endl;
	cout << "Input : " << endl;
	for (onnx::ValueInfoProto text : graph.input()) {
		cout << "	" << "Name: " << text.name() << endl;
		cout << "	" << "Type: " << text.type() << endl;
	}
	cout << endl;
	cout << "value_info : " << endl;
	for (onnx::ValueInfoProto text : graph.value_info()) {
		cout << "	" << "Name: " << text.name() << endl;
		cout << "	" << "Type: " << text.type() << endl;
	}
	cout << endl;
	cout << "Output : " << endl;
	for (onnx::ValueInfoProto text : graph.output()) {
		cout << "	" << "Name: " << text.name() << endl;
		cout << "	" << "Type: " << text.type() << endl;
	}
	cout << endl;
	for (onnx::TensorProto tensor : graph.initializer()) {
		cout << " _____ " << tensor.name() << " _____ " << endl;
		cout << "dims: ";
		for (int text : tensor.dims())
			cout << text << ", ";
		cout << endl;
		cout << "float_data : ";
		for (float text : tensor.float_data())
			cout << text << ", ";
		cout << endl;
		cout << "DataType: " << tensor.data_type() << endl;
		cout << "metadata_props: " << endl;
		for (onnx::StringStringEntryProto metadata : tensor.metadata_props()) {
			cout << "-------------------------------------------------------" << endl;
			cout << "	" << metadata.key() << ": " << metadata.value() << endl;
		}
		cout << endl;
	}
	cout << "Graph mit " << graph.node_size() << " Knoten: " << endl;
	for (onnx::NodeProto node : graph.node()) {
		cout << " _____ " << node.name() << " _____ " << endl;
		cout << "input: ";
		for (string text : node.input())
			cout << text << ", ";
		cout << endl;
		cout << "output: ";
		for (string text : node.output())
			cout << text << ", ";
		cout << endl;
		cout << "op_type: " << node.op_type() << endl;
		cout << "domain: " << node.domain() << endl;
		cout << "attribute: ";
		for (onnx::AttributeProto Attribute : node.attribute())
			cout << Attribute.name() << ", ";
		cout << endl;
		cout << "metadata_props: " << endl;
		for (onnx::StringStringEntryProto metadata : node.metadata_props()) {
			cout << "-------------------------------------------------------" << endl;
			cout << "	" << metadata.key() << ": " << metadata.value() << endl;
		}
		cout << endl;
	}
}
#include <windows.h>
#include "onnx/defs/schema.h"
int main(int argc, char *argv[]) {
	string fileName = "";
	string batchSize = "1"; // Default batch size, can be changed by user input
	std::vector<string> staticInputs;
	std::vector<string>  staticOutputs;
	cout << "\nCommand-line arguments:\n";
	int count;
	for (count = 0; count < argc; count++) {
		if (string(argv[count]) == "--file") {
			if (count + 1 < argc) {
				fileName = argv[count + 1];
				count++;
			}
			else {
				cout << "Error: --file option requires a file name argument." << endl;
				return 1;
			}
		}
		else if (string(argv[count]) == "--dynamic_dim") {
			if (count + 1 < argc) { // todo name=count 
				batchSize = argv[count + 1];
				count++;
			}
			else {
				cout << "Error: --batch_size option requires a batch size argument." << endl;
				return 1;
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
					return 1;
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
					return 1;
				}
			}
		}
	}
	fstream file;
	file.open("model.h", fstream::out | fstream::trunc); // Open file for writing. Create a new file if file with this name does not exists or clear existing file
	if (file.is_open() && file.good()) {
		try
		{
			onnx::ModelProto model;
			onnx::LoadProtoFromPath(fileName, model);
			onnx::shape_inference::DataValueMap aMap;
			onnx::shape_inference::InferShapes(model, onnx::OpSchemaRegistry::Instance(), onnx::ShapeInferenceOptions().error_mode, &aMap);
			onnx::shape_inference::SymbolTableImpl symbolTable;
			symbolTable.addFromGraph(model.graph());
			for (onnx::FunctionProto func : model.functions()) {
				cout << "Function: " << func.name() << endl;
				for (const auto& input : func.input())
				{
					cout << "Input: " << input << endl;
				}
				for (const auto& output : func.output())
				{
					cout << "Output: " << output << endl;
				}
			}
			for (onnx::OperatorSetIdProto  func : model.opset_import()) {
				cout << "OpSet: " << func.domain() << " : " << func.version() << endl;
			}
			onnx::GraphProto graphProto = model.graph();
			OnnxGraph graph(graphProto, true, staticInputs, staticOutputs, true);
			graph.PreProcess();
			file << "// Includes" << endl << endl;
			file << "#include <xtensor/xarray.hpp>" << endl;
			file << "#include <tuple>" << endl;
			file << graph.GetIncludes() << endl;
			file << graph.PrintGraph() << endl;
			//file << "std::size_t batch_size = " + batchSize + ";" << endl;   // Will be set by the user through the input args
			// file << OnnxNode::PrintPredictedDims() << endl;
			//file << join(OnnxVars::GetVarsAsStrings(vars.GetOutput()), ", ");
			//int result = std::system("clang-format -i model.h");
			file.close();
		}
		catch (const std::exception& e)
		{	
			cout << "Error: " << e.what() << endl;
		}

	}
	
	//_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
	//_CrtDumpMemoryLeaks();
	return 0;
}


