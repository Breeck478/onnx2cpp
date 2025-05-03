#include "OnnxVars.h"
//#include <onnx/onnx_pb.h>

OnnxVar::OnnxVar(onnx::ValueInfoProto valueInfo)
{
	name = valueInfo.name();
	typeProto = valueInfo.type();
}

std::string OnnxVar::GetName() const{
	return name;
}

onnx::TypeProto OnnxVar::GetTypeProto() const {
	return typeProto;
}

std::string OnnxVar::GetDataTypeString() const {
	std::string res = "";
	if (typeProto.has_tensor_type())
		res = "Tensor";
	else if (typeProto.has_sequence_type())
		res = "^Sequence";
	else if (typeProto.has_map_type())
		res = "Map";
	else if (typeProto.has_optional_type())
		res = "Optional";	
	else if (typeProto.has_sparse_tensor_type())
		res = "Sparse Tensor";
	return res;

}

std::string OnnxVar::GetVarInitString() const {
	std::string res = "";
	res += GetDataTypeString();
	res += " " + name + ";";
	return res;
}