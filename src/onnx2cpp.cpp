// onnx2cpp.cpp: Definiert den Einstiegspunkt für die Anwendung.
//
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "onnx2cpp.h"
//#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include <onnx/common/file_utils.h>
#include <onnx/defs/printer.h>
#include<onnx/shape_inference/implementation.h>
#include <vector>
#include <string>
#include <fstream>
#include "OnnxVars.h"
#include "OnnxConsts.h"
#include "OnnxNodes.h"
#include "Utils.h"
#include <cstdlib>

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
int main(){
	fstream file;
	file.open("model.h", fstream::out | fstream::trunc);
	if (file.is_open() && file.good()) {
		try
		{
			onnx::ModelProto model;
			onnx::LoadProtoFromPath("Sinus_model.onnx", model);
			onnx::shape_inference::InferShapes(model);
			onnx::GraphProto graph = model.graph();
			OnnxVars vars;
			OnnxVars input;
			OnnxVars output;
			OnnxNodes nodes;
			OnnxConsts consts;
			vars.InitWithList(graph.value_info());
			input.InitWithList(graph.input());
			output.InitWithList(graph.output(), true);
			consts.InitWithList(graph.initializer());
			nodes.InitWithGraph(graph);
			file << "// Includes" << endl << endl;
			file << "#include <vector>" << endl;
			for (const std::string type : nodes.GetOpTypes())
			{
				file << "#include <Operators/" << type << ".h>" << endl; 
			}
			file << "template <typename T>" << endl;
			file << "void model(";
			file << join(input.GetVarsAsStrings(), ", ");
			file << ", ";
			file << join(output.GetVarsAsStrings(), ", ");
			file << ") {" << endl;
			file << endl << "// Initialize variables"  << endl;
			//file << join(OnnxVars::GetVarsAsStrings(vars.GetOutput()), ", ");
			for (const std::string var : vars.GetVarsAsStrings())
				file << var << ";" << endl;
			for (const std::string var : consts.GetVarsAsStrings())
				file << var << ";" << endl;
			
			file << endl << "// Initialize nodes" << endl;
			for (const OnnxNode node : nodes)
				file << node.GetVarInitString() << endl;
			file << "}" << endl;
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


