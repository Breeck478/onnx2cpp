#pragma once
#include "OnnxVars.h"
#include "Utils.h"
#include <iostream>
#include <deque>

std::vector<std::string> OnnxVars::names;


OnnxVar::OnnxVar(onnx::ValueInfoProto valueInfo, bool isInitialising, bool isOutput)
{
	this->name = valueInfo.name();
	this->typeProto = valueInfo.type();
	this->isOutput = isOutput;
	this->is_initialized_in_model = isInitialising;
}

std::string OnnxVar::GetName() const{
	return remove_chars(name);
}

std::string OnnxVar::GetShapeName() const {
	return GetName() + "_shape"; // For xtensor shape
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
		//const auto& dims = tensorType.shape().dim();
		//if (dims.size() > 0) {
			//if (is_initialized_in_model) {
			//	res += "typename xt::xarray<T>::shape_type " + GetShapeName() + " = {";
			//	for (size_t i = 0; i < dims.size(); ++i)
			//	{
			//		if (dims[i].has_dim_value())
			//		{
			//			res += std::to_string(dims[i].dim_value());
			//		}
			//		else if (dims[i].has_dim_param())
			//		{
			//			res += dims[i].dim_param();
			//		}
			//		else
			//		{
			//			res += "0"; // Default value for unknown dimensions
			//		}
			//		if (i < dims.size() - 1)
			//			res += ", ";
			//	}
			//	res += "}; \n";
			//}

			res += "xt::xarray<";
			res += "T>";
			if (isOutput)
				res += "&";
			res += " " + GetName();
			//if (is_initialized_in_model) {
			//	res += "(" + GetShapeName() + ", 0.0)"; // Initialize with zeros
			//}
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

std::string OnnxVar::GetVariableString() {
	std::string res = "";
	res += GetDataTypeString();
	return res;
}
bool OnnxVar::SetInitialization() {
	const onnx::TypeProto_Tensor tensorType = typeProto.tensor_type();
	const auto& dims = tensorType.shape().dim();
	for (auto dim : dims)
	{
		if (dim.has_dim_param() && dim.dim_param() != "batch_size")
		{
			SetContainsUnkownDim();
			return true;
		}
	}
	return false;
}

// Vars
void OnnxVars::InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list, bool isInitialising, bool isOutput){
	Clear();
	for (onnx::ValueInfoProto valueInfo : list) {
		Add(OnnxVar(valueInfo, isInitialising, isOutput));
	}

}

void OnnxVars::SetInitializations() {
	for (OnnxVar& var : vars) {
		var.SetInitialization();
	}
}

void OnnxVars::Add(const OnnxVar var) {
	std::string name = remove_chars(var.GetName());
	if ((names.end() == std::find(names.begin(), names.end(), name))) {
		names.push_back(name);
		vars.push_back(var);
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
	for (OnnxVar var : vars)
	{
		if (var.ContainsUnkownDim()) {
			continue; // Skip variables that are initialized by operators
		}
		std::string varString = var.GetVariableString();
		if (!varString.empty()) {
			res.push_back(varString); // Add semicolon to the end of the variable declaration
		}
		
	}
	return res;
}

OnnxVar& OnnxVars::operator[](int i) {
	return vars[i];
}
std::deque<OnnxVar>::const_iterator OnnxVars::begin() const {
	return vars.begin();
}
std::deque<OnnxVar>::const_iterator OnnxVars::end() const {
	return vars.end();
}
std::deque<OnnxVar>::iterator OnnxVars::begin() {
	return vars.begin();
}
std::deque<OnnxVar>::iterator OnnxVars::end() {
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

bool OnnxVars::FindConstPointerByName(const std::string name, OnnxVar*& OutputVar) const {
	for (const OnnxVar& c : vars) {
		if (c.GetName() == name) {
			OutputVar = const_cast<OnnxVar*>(&c);
			return true; // Found the const
		}
	}
	return false;
}