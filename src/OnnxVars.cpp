#include "OnnxVars.h"
#include "Utils.h"
#include <algorithm>
#include <iostream>


std::vector<std::string> OnnxVars::names;

OnnxVar::OnnxVar(onnx::ValueInfoProto valueInfo)
{
	this->name = valueInfo.name();
	this->typeProto = valueInfo.type();
}

std::string OnnxVar::GetName() const{
	return remove_chars(name);
}

onnx::TypeProto OnnxVar::GetTypeProto() const {
	return typeProto;
}


std::string OnnxVar::GetDataTypeString() const {
	std::string res = "";
	if (typeProto.has_tensor_type())
	{
		const onnx::TypeProto_Tensor tensorType = typeProto.tensor_type();
		//res = Utils::GetDataTypeString(tensorType.elem_type());
		res = "";// "std::vector<T> const& " + name;
		const auto& dims = tensorType.shape().dim();
		for (size_t i = 0; i < dims.size(); ++i)
		{
			res += "std::vector<";
		}
		res += "T";
		for (size_t i = 0; i < dims.size(); ++i)
		{
			res += ">";
		}
		res += " " + GetName();
		//for (const auto& dim : dims)
		//{
		//	if (dim.has_dim_value())
		//	{
		//		res += "[" + std::to_string(dim.dim_value()) + "]";
		//	}
		//	else if (dim.has_dim_param())
		//	{
		//		res += "[" + dim.dim_param() + "]";
		//	}
		//	else
		//	{
		//		res += "[]";
		//	}
		//}
	}
	else if (typeProto.has_sequence_type())
		res = "Sequence";
	else if (typeProto.has_map_type())
		res = "Map";
	else if (typeProto.has_optional_type())
		res = "Optional";
	else if (typeProto.has_sparse_tensor_type())
	{
		res = "Sparse Tensor";
		const auto& dims = typeProto.tensor_type().shape().dim();
		for (const auto& dim : dims)
		{
			if (dim.has_dim_value())
			{
				res += "[" + std::to_string(dim.dim_value()) + "]";
			}
			else if (dim.has_dim_param())
			{
				res += "[" + dim.dim_param() + "]";
			}
			else
			{
				res += "[]";
			}
		}
	}
	return res;
}

std::string OnnxVar::GetVarInitString() const {
	std::string res = "";
	res += GetDataTypeString();
	return res;
}

// Vars
void OnnxVars::InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list){
	Clear();

	for (onnx::ValueInfoProto valueInfo : list) {
		Add(OnnxVar(valueInfo), vars);
	}

}

void OnnxVars::Add(const OnnxVar var, std::vector<OnnxVar> &list) {
	list.push_back(var);
	std::string name = remove_chars(var.GetName());
	if ((names.end() == std::find(names.begin(), names.end(), name))) {
		names.push_back(name);
	}
	else {
		std::cout << "var " << var.GetName() << " is already added" << std::endl; // Can´t happen. ERROR
	}
}
int OnnxVars::GetCount() const {
	return vars.size();
}
const OnnxVar& OnnxVars::operator[](int i) const {
	return vars[i];
}
std::vector<std::string> OnnxVars::GetVarsAsStrings() {
	std::vector<std::string> res;
	for (const OnnxVar var : vars)
	{
		res.push_back(var.GetDataTypeString());
	}
	return res;
}

OnnxVar& OnnxVars::operator[](int i) {
	return vars[i];
}
std::vector<OnnxVar>::const_iterator OnnxVars::begin() const {
	return vars.begin();
}
std::vector<OnnxVar>::const_iterator OnnxVars::end() const {
	return vars.end();
}
std::vector<OnnxVar>::iterator OnnxVars::begin() {
	return vars.begin();
}
std::vector<OnnxVar>::iterator OnnxVars::end() {
	return vars.end();
}

// names

std::string OnnxVars::GetName(const int i) const {
	return names[i];
}
std::vector<std::string> OnnxVars::GetNames() const {
	return names;
}
int OnnxVars::GetNameCount() const {
	return names.size();
}