//#include <iostream>
#include <google/protobuf/text_format.h>
#include <onnx/onnx_pb.h>



onnx::GraphProto* getGraph(onnx::ModelProto* model);

onnx::TensorProto* getTensor(onnx::GraphProto* graph, int index);

onnx::NodeProto* getNode(onnx::GraphProto* graph, int index);

onnx::AttributeProto* getAttribute(onnx::NodeProto* node, int index);



