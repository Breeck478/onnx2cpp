#pragma once
#include "OnnxConst.h"
#include "Utils.h"
#include <algorithm>
#include <iostream>
std::vector<std::string> OnnxConsts::names;
OnnxConst::OnnxConst(onnx::TensorProto tensorProto)
{
	this->name = tensorProto.name();
	this->Shape(tensorProto.dims());
	this->dataType = tensorProto.data_type();
	this->FillData(tensorProto);
}

void OnnxConst::FillData(const onnx::TensorProto& tensorProto) {
	if (DataType() == onnx::TensorProto_DataType_FLOAT) { // complex not supported
		if (tensorProto.float_data().size() > 0)
			this->data = ParseRepeatedField(tensorProto.float_data());
		else
			this->data = ParseByteData<float>(tensorProto.raw_data());
	} else if (DataType() == onnx::TensorProto_DataType_INT32 || DataType() == onnx::TensorProto_DataType_INT16 || DataType() == onnx::TensorProto_DataType_INT8 || DataType() == onnx::TensorProto_DataType_UINT32 || DataType() == onnx::TensorProto_DataType_UINT16 || DataType() == onnx::TensorProto_DataType_UINT8 || DataType() == onnx::TensorProto_DataType_BOOL) {  // int4, uint4, FLOAT16, BFLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ, FLOAT4E2M1 not supported
		if (tensorProto.int32_data().size() > 0)
			this->data = ParseRepeatedField(tensorProto.int32_data());
		else
			this->data = ParseByteData<int32_t>(tensorProto.raw_data());
	} else if (DataType() == onnx::TensorProto_DataType_STRING) {
		if (tensorProto.string_data().size() > 0)
			this->data = ParseRepeatedField(tensorProto.string_data());
		else
			this->data = ParseByteData<std::string>(tensorProto.raw_data());
	} else if (DataType() == onnx::TensorProto_DataType_INT64) {
		if (tensorProto.int64_data().size() > 0)
			this->data = ParseRepeatedField(tensorProto.int64_data());
		else
			this->data = ParseByteData<int64_t>(tensorProto.raw_data());
	}
	else {
		std::runtime_error("ERROR(OnnxConst::FillData): Tensor data type not supported for Constante " + Name());
	}
}

void OnnxConst::Shape(::google::protobuf::RepeatedField<int64_t> dims) {
	this->shape.clear();
	this->shape.reserve(dims.size());
	for (const auto& dim : dims) {
		this->shape.push_back(dim);
	}
}

std::string OnnxConst::GetShapeName() const {
	return Name() + "_shape"; // For xtensor shape
}

std::vector<std::any> OnnxConst::GetDataAsAny() const {
	return std::visit([](const auto& vec) -> std::vector<std::any> {
		std::vector<std::any> result;
		for (const auto& v : vec) {
			result.push_back(v);
		}
		return result;
		}, data);
}

OnnxConst::TensorData OnnxConst::GetData() const {
	return data;
}

size_t OnnxConst::GetDataSize() const {
	return std::visit([](auto&& arg) -> size_t {
		return arg.size();
		}, data);
}

template <typename T>
std::vector<T> OnnxConst::GetDataAsT() const {
	if (!std::holds_alternative<std::vector<T>>(data)) {
		std::runtime_error("ERROR(OnnxConst::GetDataAsT): Tensor data type not supported for Constant " + Name());
		return {};
	}
	return std::get<std::vector<T>>(data);
}

template <typename T>
std::string OnnxConst::GenerateNestedInitializerFromAny() const {
	std::ostringstream oss;
	oss << "{";
	std::vector<std::any> vals = GetDataAsAny();
	for (int64_t i = 0; i < vals.size(); ++i) {
		const T& val = std::any_cast<T>(vals[i]);
		if constexpr (std::is_floating_point_v<T>) {
			oss << std::fixed << std::setprecision(20) << val << "f";
		}
		else {
			oss << val;
		}
		if (i + 1 < vals.size()) oss << ", ";
	}
	oss << "}";
	return oss.str();
}
std::string OnnxConst::PrintShape() {
	std::string res = "";
	if (shape.size() > 0) {
		std::vector<int64_t> shape(shape.begin(), shape.end());
		res += "typename xt::xarray<" + GetDataTypeAsString() + ">::shape_type " + GetShapeName() + " = {";
		for (size_t i = 0; i < shape.size(); ++i) {
			if (shape[i] < 0) {
				std::cout << "ERROR: Negative dimension in tensor shape" << std::endl;
				return res;
			}
			if (i > 0) res += ", ";
			res += std::to_string(shape[i]);
		}
		res += "};\n"; // Initialize with zeros
	}
	return res;
}

std::string OnnxConst::PrintReshape() {
	std::string res = "";
	if (shape.size() > 0) {
		res += "\n" + Name() + " = " + Name() + ".reshape(" + GetShapeName() + ");";
	}
	return res;
}

std::string OnnxConst::GetDataTypeString(bool const doInitialize) {
	std::string res = "";
	if (GetDataSize() > 0) {

		std::ostringstream oss;
		if (doInitialize) 
			oss << "xt::xarray<" + GetDataTypeAsString() + "> ";

		oss << Name() << " = ";

		if (std::holds_alternative<std::vector<float>>(data)) {
			oss << GenerateNestedInitializerFromAny<float>();
		}
		else if (std::holds_alternative<std::vector<int32_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int32_t>();
		}
		else if (std::holds_alternative<std::vector<int16_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int16_t>();
		}
		else if (std::holds_alternative<std::vector<int8_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int8_t>();
		}
		else if (std::holds_alternative<std::vector<uint8_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<uint8_t>();
		}
		else if (std::holds_alternative<std::vector<uint16_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<uint16_t>();
		}
		else if (std::holds_alternative<std::vector<uint32_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<uint32_t>();
		}
		else if (std::holds_alternative<std::vector<bool>>(data)) {
			oss << GenerateNestedInitializerFromAny<bool>();
		}
		else if (std::holds_alternative<std::vector<int64_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int64_t>();
		}
		else if (std::holds_alternative<std::vector<double>>(data)) {
			oss << GenerateNestedInitializerFromAny<double>();
		}
		else if (std::holds_alternative<std::vector<std::string>>(data)) {
			oss << GenerateNestedInitializerFromAny<std::string>();
			//GetDataAsT<std::string>();
		}
		else if (std::holds_alternative<std::vector<uint64_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<uint64_t>();

		}
		else {
			std::cout << "ERROR: Tensor data type not supported" << std::endl;
		}
		res += oss.str() + ";";
	}
	return res;
}

std::string OnnxConst::GetConstantString(bool const doInitialize) {
	std::string res = "";
	res += PrintShape();
	res += GetDataTypeString(doInitialize);
	res += PrintReshape();
	return res;
}

void OnnxConst::PreProcess() {

}

// Vars
void OnnxConsts::InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto>& list){
	Clear();
	for (onnx::TensorProto tensorProto : list) {
		Add(OnnxConst(tensorProto));
	}

}
void OnnxConsts::Add(const OnnxConst var) {
	std::string name = remove_chars(var.Name());
	if ((names.end() == std::find(names.begin(), names.end(), name))) {
		names.push_back(name);
		vars.push_back(var);
	}
	else {
		std::cout << "var " << var.Name() << " is already added" << std::endl; // Can´t happen. ERROR
	}
}


int OnnxConsts::GetCount() const {
	return vars.size();
}
const OnnxConst& OnnxConsts::operator[](int i) const {
	return vars[i];
}
std::vector<std::string> OnnxConsts::GetVarsAsStrings() const {
	std::vector<std::string> res;
	for (OnnxConst var : vars)
	{
		res.push_back(var.GetConstantString());
	}
	return res;
}

OnnxConst& OnnxConsts::operator[](int i) {
	return vars[i];
}
std::deque<OnnxConst>::const_iterator OnnxConsts::begin() const {
	return vars.begin();
}
std::deque<OnnxConst>::const_iterator OnnxConsts::end() const {
	return vars.end();
}
std::deque<OnnxConst>::iterator OnnxConsts::begin() {
	return vars.begin();
}
std::deque<OnnxConst>::iterator OnnxConsts::end() {
	return vars.end();
}

// names

std::string OnnxConsts::GetName(const int i) const {
	return names[i];
}
std::vector<std::string> OnnxConsts::GetNames() const {
	return names;
}
int OnnxConsts::GetNameCount() const {
	return names.size();
}

