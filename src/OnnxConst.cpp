#include "OnnxConst.h"
#include "Utils.h"
#include <algorithm>
#include <iostream>
using namespace toCpp;
OnnxConst::OnnxConst(onnx::TensorProto &tensorProto)
{
	this->name = tensorProto.name();
	this->Shape(tensorProto.dims());
	this->dataType = tensorProto.data_type();
	this->FillData(tensorProto);
}

void OnnxConst::FillData(const onnx::TensorProto& tensorProto) {
	switch (DataType()) {
	case (onnx::TensorProto_DataType_FLOAT):
		this->data = ExtractDataFromTensor<float>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_INT64):
		this->data = ExtractDataFromTensor<int64_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_INT32):
		this->data = ExtractDataFromTensor<int32_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_INT16):
		this->data = ExtractDataFromTensor<int16_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_INT8):
		this->data = ExtractDataFromTensor<int8_t>(tensorProto);

		break;
	case (onnx::TensorProto_DataType_UINT64):
		this->data = ExtractDataFromTensor<uint64_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_UINT32):
		this->data = ExtractDataFromTensor<uint32_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_UINT16):
		this->data = ExtractDataFromTensor<uint16_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_UINT8):
		this->data = ExtractDataFromTensor<uint8_t>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_DOUBLE):
		this->data = ExtractDataFromTensor<double>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_STRING):
		this->data = ExtractDataFromTensor<std::string>(tensorProto);
		break;
	case (onnx::TensorProto_DataType_BOOL):
		this->data = ExtractDataFromTensor<bool>(tensorProto);
		break;
	default:
		throw std::runtime_error("ERROR(OnnxConst::FillData): Tensor data type " + GetDataTypeString(DataType()) + " not supported for Constant " + Name());


	}
}

void OnnxConst::Shape(::google::protobuf::RepeatedField<int64_t> dims) {
	this->shape.clear();
	this->shape.reserve(dims.size());
	for (const auto dim : dims) {
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

// can be used to get information like size without casting the data
template <typename T>
std::vector<T> OnnxConst::GetDataAsT() const {
	if (!std::holds_alternative<std::vector<T>>(data)) {
		throw std::runtime_error("ERROR(OnnxConst::GetDataAsT): Tensor data type not supported for Constant " + Name());
		return {};
	}
	return std::get<std::vector<T>>(data);
}

std::string OnnxConst::PrintShape() {
	std::string res = "";
	if (shape.size() > 0) {
		std::vector<int64_t> shapeVector(shape.begin(), shape.end());
		res += "typename xt::xarray<" + GetDataTypeAsString() + ">::shape_type " + GetShapeName() + " = {";
		for (size_t i = 0; i < shapeVector.size(); ++i) {
			if (shapeVector[i] < 0) {
				std::cout << "ERROR: Negative dimension in tensor shape" << std::endl;
				return res;
			}
			if (i > 0) res += ", ";
			res += std::to_string(shapeVector[i]);
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

std::string OnnxConst::GetDataAsString(bool const doInitialize) {
	std::string res = "";
	if (GetDataSize() > 0) {

		std::ostringstream oss;
		if (doInitialize) 
			oss << "xt::xarray<" + GetDataTypeAsString() + "> ";

		oss << Name() << " = ";

		if (std::holds_alternative<std::vector<float>>(data)) {
			oss << "{" << Join(std::get<std::vector<float>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<bool>>(data)) {
			oss << "{" << Join(std::get<std::vector<bool>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<int32_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<int32_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<int16_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<int16_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<int8_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<int8_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<uint8_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<uint8_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<uint16_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<uint16_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<uint32_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<uint32_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<int64_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<int64_t>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<double>>(data)) {
			oss << "{" << Join(std::get<std::vector<double>>(data), ", ") << "}";
		}
		else if (std::holds_alternative<std::vector<std::string>>(data)) {
			oss << "{\"" << Join(std::get<std::vector<std::string>>(data), "\", \"") << "\"}";
		}
		else if (std::holds_alternative<std::vector<uint64_t>>(data)) {
			oss << "{" << Join(std::get<std::vector<uint64_t>>(data), ", ") << "}";
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
	if (GetDataSize() <= 0) {
		res += "xt::xarray<" + GetDataTypeAsString() + "> " + Name() + " = xt::empty<" + GetDataTypeAsString() + ">(" + GetShapeName() + ");\n"; // Initialize with zeros
		return res;
	}
	res += GetDataAsString(doInitialize);
	res += PrintReshape();
	return res;
}

// Vars
void OnnxConsts::InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto>& list){
	Clear();
	for (onnx::TensorProto tensorProto : list) {
		Add(OnnxConst(tensorProto));
	}

}
void OnnxConsts::Add(const OnnxConst constant) {
	// make sure it is not already added
	if ((consts.end() == std::find(consts.begin(), consts.end(), constant))) {
		consts.push_back(constant);
	}
	else {
		std::cout << "var " << constant.Name() << " is already added" << std::endl; // Can´t happen. ERROR
	}
}


int OnnxConsts::GetCount() const {
	return consts.size();
}
const OnnxConst& OnnxConsts::operator[](int i) const {
	return consts[i];
}
std::vector<std::string> OnnxConsts::GetConstsAsStrings() const {
	std::vector<std::string> res;
	for (OnnxConst var : consts)
	{
		res.push_back(var.GetConstantString());
	}
	return res;
}

OnnxConst& OnnxConsts::operator[](int i) {
	return consts[i];
}
std::vector<OnnxConst>::const_iterator OnnxConsts::begin() const {
	return consts.begin();
}
std::vector<OnnxConst>::const_iterator OnnxConsts::end() const {
	return consts.end();
}
std::vector<OnnxConst>::iterator OnnxConsts::begin() {
	return consts.begin();
}
std::vector<OnnxConst>::iterator OnnxConsts::end() {
	return consts.end();
}

bool OnnxConsts::FindConstPointerByName(const std::string name, OnnxConst*& outputConst) const {
	for (const OnnxConst& c : consts) {
		if (c.Name() == name) {
			outputConst = const_cast<OnnxConst*>(&c);
			return true; // Found the const
		}
	}
	return false;
}

