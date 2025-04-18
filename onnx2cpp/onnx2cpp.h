#include "onnx/onnx.pb.h"


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

onnx::TensorProto* getInput(onnx::GraphProto* graph, int index) {
	if (graph == nullptr) {
		return nullptr;
	}
	if (index < 0 || index >= graph->input_size()) {
		return nullptr;
	}
	return graph->mutable_input(index);
}

onnx::TensorProto* getOutput(onnx::GraphProto* graph, int index) {
	if (graph == nullptr) {
		return nullptr;
	}
	if (index < 0 || index >= graph->output_size()) {
		return nullptr;
	}
	return graph->mutable_output(index);
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

onnx::TensorProto_DataType getDataType(onnx::TensorProto* tensor) {
	if (tensor == nullptr) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	return tensor->data_type();
}

onnx::TensorProto_DataType getDataType(onnx::AttributeProto* attribute) {
	if (attribute == nullptr) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	return attribute->t().data_type();
}

onnx::TensorProto_DataType getDataType(onnx::NodeProto* node, int index) {
	if (node == nullptr) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	if (index < 0 || index >= node->attribute_size()) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	return node->attribute(index).t().data_type();
}

onnx::TensorProto_DataType getDataType(onnx::GraphProto* graph, int index) {
	if (graph == nullptr) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	if (index < 0 || index >= graph->initializer_size()) {
		return onnx::TensorProto_DataType_UNDEFINED;
	}
	return graph->initializer(index).data_type();
}
#include <iostream>
int main(int argc, char* argv[]);
	onnx::ModelProto model;
	model = onnx::loadModel("model.onnx");
	onnx::GraphProto* graph = getGraph(&model);
	if (graph == nullptr) {
		cout << "Graph is null" << endl;
		return -1;
	}

	onnx::TensorProto* tensor = getTensor(graph, 0);
	if (tensor == nullptr) {
		cout << "Tensor is null" << endl;
		return -1;
	}

	onnx::TensorProto_DataType dataType = getDataType(tensor);
	cout << "Data type: " << dataType << endl;
	return 0;
}
