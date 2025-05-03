// onnx2cpp.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "onnx2cpp.h"
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>

#include <onnx/common/file_utils.h>
#include <onnx/defs/printer.h>
#include <vector>
#include <string>
#include <fstream>
#include "OnnxVars.h"


using namespace std;

void PrintGraphInfo(onnx::GraphProto graph) {
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

void collectAllVars(onnx::GraphProto, vector<OnnxVar> vars) {

}

int main(){
	fstream file;
	file.open("model.cpp", fstream::out | fstream::trunc);
	onnx::ModelProto model;
	onnx::LoadProtoFromPath("model.onnx", model);
	onnx::GraphProto graph = model.graph();
	//PrintGraphInfo(graph);
	vector<OnnxVar> vars;
	for (OnnxVar info : graph.input()) {
		vars.push_back(info);
	}
	for (OnnxVar info : graph.value_info()) {
		vars.push_back(info);
	}
	for (OnnxVar info : graph.output()) {
		vars.push_back(info);
	}
	for (const OnnxVar var : vars)
		file << var.GetVarInitString() << endl;
	file.close();
	return 0;
}


