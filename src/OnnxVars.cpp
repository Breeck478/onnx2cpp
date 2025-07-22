#pragma once
#include "OnnxVars.h"
#include "Utils.h"
#include <iostream>
#include <deque>

// std::vector<std::string> OnnxVars::names;


OnnxVar::OnnxVar(onnx::ValueInfoProto valueInfo, bool isInput, bool isOutput) : isInput(isInput), isOutput(isOutput){
	this->name = valueInfo.name();
	auto typeProto = valueInfo.type();
	if (typeProto.has_tensor_type()) {
		this->dataType = typeProto.tensor_type().elem_type();
		this->Shape(typeProto.tensor_type().shape());
	}
	else if (typeProto.has_sparse_tensor_type()) {
		this->dataType = typeProto.sparse_tensor_type().elem_type();
		this->Shape(typeProto.sparse_tensor_type().shape());
	}
	else {
		throw std::runtime_error("ERROR(OnnxVar::OnnxVar): TypeProto-Type not supported yet for Var" + valueInfo.name());
	}
}

void OnnxVar::Shape(onnx::TensorShapeProto shapeProto) {
	auto dims = shapeProto.dim();
	this->shape.clear();
	this->shape.reserve(dims.size());
	for (const auto& dim : dims) {
		if (dim.has_dim_value()) {
			this->shape.push_back(dim.dim_value());
		}
		else if (dim.has_dim_param()) {
			if (dim.dim_param() == "batch_size") { 
				this->shape.push_back(OnnxTensor::batchSize); 
			}
			else {
				this->shape.push_back(-1);
				SetContainsUnkownDim();
			}
		}
		else {
			// unbekannte dimension ohne zusammenhang zu einer anderen Dimension bzw einem anderem Tensor siehe https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes
			//throw std::runtime_error("ERROR(OnnxVar::Shape): no specified value type for Var" + Name());
		}
	}
}

std::string OnnxVar::GetShapeName() const {
	return Name() + "_shape"; // For xtensor shape
}



//std::string OnnxVar::GetDataTypeString() const {
//	std::string res = "";
//	if (typeProto.has_tensor_type())
//	{
//		const onnx::TypeProto_Tensor tensorType = typeProto.tensor_type();
//		//res = Utils::GetDataTypeString(tensorType.elem_type());
//		res = "";// "std::vector<T> const& " + name;
//		//const auto& dims = tensorType.shape().dim();
//		//if (dims.size() > 0) {
//			//if (is_initialized_in_model) {
//			//	res += "typename xt::xarray<T>::shape_type " + GetShapeName() + " = {";
//			//	for (size_t i = 0; i < dims.size(); ++i)
//			//	{
//			//		if (dims[i].has_dim_value())
//			//		{
//			//			res += std::to_string(dims[i].dim_value());
//			//		}
//			//		else if (dims[i].has_dim_param())
//			//		{
//			//			res += dims[i].dim_param();
//			//		}
//			//		else
//			//		{
//			//			res += "0"; // Default value for unknown dimensions
//			//		}
//			//		if (i < dims.size() - 1)
//			//			res += ", ";
//			//	}
//			//	res += "}; \n";
//			//}
//
//			res += "xt::xarray<";
//			res += "T>";
//			if (isOutput)
//				res += "&";
//			res += " " + Name();
//			//if (is_initialized_in_model) {
//			//	res += "(" + GetShapeName() + ", 0.0)"; // Initialize with zeros
//			//}
//			//for (const auto& dim : dims)
//			//{
//			//	if (dim.has_dim_value())
//			//	{
//			//		res += "[" + std::to_string(dim.dim_value()) + "]";
//			//	}
//			//	else if (dim.has_dim_param())
//			//	{
//			//		res += "[" + dim.dim_param() + "]";
//			//	}
//			//	else
//			//	{
//			//		res += "[]";
//			//	}
//			//}
//		}
//		else if (typeProto.has_sequence_type())
//			res = "Sequence";
//		else if (typeProto.has_map_type())
//			res = "Map";
//		else if (typeProto.has_optional_type())
//			res = "Optional";
//		else if (typeProto.has_sparse_tensor_type())
//	{
//		res = "Sparse Tensor";
//		const auto& dims = typeProto.tensor_type().shape().dim();
//		for (const auto& dim : dims)
//		{
//			if (dim.has_dim_value())
//			{
//				res += "[" + std::to_string(dim.dim_value()) + "]";
//			}
//			else if (dim.has_dim_param())
//			{
//				res += "[" + dim.dim_param() + "]";
//			}
//			else
//			{
//				res += "[]";
//			}
//		}
//	}
//	
//
//	return res;
//}

std::string OnnxVar::GetVariableString() {
	std::string res = "";
	res += "xt::xarray<" + GetDataTypeAsString() + "> ";
	if (IsIO() && IsOutput())
		res += "&";
	res += Name();
	if (!IsIO())
		res += ";";
	return res;
}

void OnnxVar::PreProcess() {

}



// Vars
void OnnxVars::AddFromList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list, bool isInput, bool isOutput, bool isInitialising){
	
	for (onnx::ValueInfoProto valueInfo : list) {
		Add(OnnxVar(valueInfo, isInput, isOutput, isInitialising));
	}

}


void OnnxVars::Add(const OnnxVar var) {
	std::string name = remove_chars(var.Name());
	if ((names.end() == std::find(names.begin(), names.end(), name))) {
		names.push_back(name);
		vars.push_back(var);
	}
	else {
		std::cout << "var " << var.Name() << " is already added" << std::endl; // Can´t happen. ERROR
	}
}
int OnnxVars::GetCount() const {
	return vars.size();
}
const OnnxVar& OnnxVars::operator[](int i) const {
	return vars[i];
}
std::vector<std::string> OnnxVars::GetVarsAsStrings() const {
	std::vector<std::string> res;
	for (OnnxVar var : vars)
	{	
		if (var.IsIO() || !var.NeedsInit()) {
			// Skip input and output variables, they are not initialized in the model
			continue;
		}
		//if (var.ContainsUnkownDim()) {
		//	continue; // Skip variables that are initialized by operators
		//}
		std::string varString = var.GetVariableString();
		if (!varString.empty()) {
			res.push_back(varString); // Add semicolon to the end of the variable declaration
		}
		
	}
	return res;
}

std::vector<std::string> OnnxVars::GetIOsAsStrings() const {
	std::vector<std::string> res;
	for (OnnxVar var : vars)
	{
		if (!var.IsIO() || !var.NeedsInit()) {
			continue;
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

bool OnnxVars::FindVarPointerByName(const std::string name, OnnxVar*& OutputVar) const {
	for (const OnnxVar& c : vars) {
		if (c.Name() == name) {
			OutputVar = const_cast<OnnxVar*>(&c);
			return true; // Found the const
		}
	}
	return false;
}

std::vector<OnnxVar*> OnnxVars::GetInputVars() const {
	std::vector<OnnxVar*> inputVars;
	for (const OnnxVar& var : vars) {
		if (var.IsInput()) {
			inputVars.push_back(const_cast<OnnxVar*>(&var));
		}
	}
	return inputVars;
}
std::vector<OnnxVar*> OnnxVars::GetOutputVars() const {
	std::vector<OnnxVar*> outputVars;
	for (const OnnxVar& var : vars) {
		if (var.IsOutput()) {
			outputVars.push_back(const_cast<OnnxVar*>(&var));
		}
	}
	return outputVars;
}