// onnx2cpp.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "onnx2cpp.h"
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>
#include <iostream>
#include <fstream>

using namespace std;

onnx::GraphProto* getGraph(onnx::ModelProto* model) {
	if (model == nullptr) {
		return nullptr;
	}
	return model->mutable_graph();
}

onnx::TensorProto* getTensor(onnx::GraphProto* graph, int index) {
	if (graph == nullptr) {
		return nullptr;
	}
	if (index < 0 || index >= graph->initializer_size()) {
		return nullptr;
	}
	return graph->mutable_initializer(index);
}
onnx::NodeProto* getNode(onnx::GraphProto* graph, int index) {
	if (graph == nullptr) {
		return nullptr;
	}
	if (index < 0 || index >= graph->node_size()) {
		return nullptr;
	}
	return graph->mutable_node(index);
}

onnx::AttributeProto* getAttribute(onnx::NodeProto* node, int index) {
	if (node == nullptr) {
		return nullptr;
	}
	if (index < 0 || index >= node->attribute_size()) {
		return nullptr;
	}
	return node->mutable_attribute(index);
}

int main(){
	std::ifstream input("model.onnx");
	onnx::ModelProto model;
	model.ParseFromIstream(&input);
	onnx::GraphProto* graph = getGraph(&model);
	if (graph == nullptr) {
		cout << "Graph is null" << endl;
		return -1;
	}else {
		cout << "Graph: " << graph << endl;
	}

	onnx::TensorProto* tensor = getTensor(graph, 0);
	if (tensor == nullptr) {
		cout << "Tensor is null" << endl;
		return -1;
	}
	else{
		cout << "Tensor: " << tensor << endl;
	}
	return 0;
}
