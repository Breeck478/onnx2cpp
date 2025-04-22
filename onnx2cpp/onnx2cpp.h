//#include <iostream>
#include <onnx/onnx_pb.h>
#include <google/protobuf/text_format.h>


onnx::GraphProto* getGraph(onnx::ModelProto* model);

onnx::TensorProto* getTensor(onnx::GraphProto* graph, int index);

onnx::TensorProto* getInput(onnx::GraphProto* graph, int index);

onnx::TensorProto* getOutput(onnx::GraphProto* graph, int index);

onnx::NodeProto* getNode(onnx::GraphProto* graph, int index);

onnx::AttributeProto* getAttribute(onnx::NodeProto* node, int index);

onnx::TensorProto_DataType getDataType(onnx::TensorProto* tensor);

onnx::TensorProto_DataType getDataType(onnx::AttributeProto* attribute);

onnx::TensorProto_DataType getDataType(onnx::NodeProto* node, int index);

onnx::TensorProto_DataType getDataType(onnx::GraphProto* graph, int index);

int main(int argc, char* argv[]);

