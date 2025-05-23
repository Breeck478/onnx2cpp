#include "OnnxConsts.h"
#include "Utils.h"
#include <algorithm>
#include <iostream>


std::vector<std::string> OnnxConsts::names;

OnnxConst::OnnxConst(onnx::TensorProto tensorProto)
{
	this->name = tensorProto.name();
	this->dims = tensorProto.dims();
	if (tensorProto.data_type() == onnx::TensorProto_DataType_FLOAT) {
		
		this->data = ParseRawData<float>(tensorProto);
	}
	else if (tensorProto.data_type() == onnx::TensorProto_DataType_INT32) {
		this->data = ParseRawData<int32_t>(tensorProto);
	}
	else if (tensorProto.data_type() == onnx::TensorProto_DataType_INT64) {
		this->data = ParseRawData<int64_t>(tensorProto);
	}
	else if (tensorProto.data_type() == onnx::TensorProto_DataType_DOUBLE) {
		this->data = ParseRawData<double>(tensorProto);
	}
	else if (tensorProto.data_type() == onnx::TensorProto_DataType_STRING) {
		this->data = ParseRawData<std::string>(tensorProto);
	}
	else {
		std::cout << "ERROR: Tensor data type not supported" << std::endl;
	}
}

std::string OnnxConst::GetName() const{
	return name;
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

const ::google::protobuf::RepeatedField<int64_t> OnnxConst::GetDims() const {
	return dims;
}

OnnxConst::TensorData OnnxConst::GetData() const {
	return data;
}

template <typename T>
std::string OnnxConst::GenerateNestedInitializerFromAny(const std::vector<int64_t>& shape,	size_t& offset,	int level) const {
	std::ostringstream oss;
	oss << "{";
	std::vector<std::any> vals = GetDataAsAny();
	int64_t dim = shape[level];
	for (int64_t i = 0; i < dim; ++i) {
		if (level + 1 == shape.size()) {
			// Leaf level – cast element from std::any
			if (offset >= vals.size()) {
				throw std::runtime_error("Zu wenige Daten vorhanden.");
			}

			try {
				const T& val = std::any_cast<T>(vals[offset++]);
				if constexpr (std::is_floating_point_v<T>) {
					oss << std::fixed << std::setprecision(6) << val << "f";
				}
				else {
					oss << val;
				}
			}
			catch (const std::bad_any_cast& e) {
				throw std::runtime_error("std::any_cast fehlgeschlagen: falscher Typ?");
			}
		}
		else {
			// Rekursion
			oss << GenerateNestedInitializerFromAny<T>(shape, offset, level + 1);
		}

		if (i + 1 < dim) oss << ", ";
	}

	oss << "}";
	return oss.str();
}

std::string OnnxConst::GetDataTypeString() const {
	std::string res = "";
	//res = Utils::GetDataTypeString(tensorType.elem_type());
	res = "";// "std::vector<T> const& " + name;
	for (size_t i = 0; i < dims.size(); ++i)
	{
		res += "std::vector<";
	}
	res += "T";
	for (size_t i = 0; i < dims.size(); ++i)
	{
		res += ">";
	}
	res += " " + remove_chars(name) + " = ";
	size_t offset = 0;
	std::ostringstream oss;
	std::vector<int64_t> shape(dims.begin(), dims.end());

	if (std::holds_alternative<std::vector<float>>(data)) {
		oss << GenerateNestedInitializerFromAny<float>(shape, offset);
	}
	else if (std::holds_alternative<std::vector<int32_t>>(data)) {
		oss << GenerateNestedInitializerFromAny<int32_t>(shape, offset);
	}
	else if (std::holds_alternative<std::vector<int64_t>>(data)) {
		oss << GenerateNestedInitializerFromAny<int64_t>(shape, offset);
	}
	else if (std::holds_alternative<std::vector<double>>(data)) {
		oss << GenerateNestedInitializerFromAny<double>(shape, offset);
	}
	else if (std::holds_alternative<std::vector<std::string>>(data)) {
		oss << GenerateNestedInitializerFromAny<std::string>(shape, offset);
	}
	else if (std::holds_alternative<std::vector<uint64_t>>(data)) {
		oss << GenerateNestedInitializerFromAny<uint64_t>(shape, offset);
	}
	else {
		std::cout << "ERROR: Tensor data type not supported" << std::endl;
	}
	res += oss.str();
	return res;
}

std::string OnnxConst::GetVarInitString() const {
	std::string res = "";
	res += GetDataTypeString();
	return res;
}

// Vars
void OnnxConsts::InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto>& list){
	Clear();

	for (onnx::TensorProto tensorProto : list) {
		Add(OnnxConst(tensorProto), vars);
	}

}

void OnnxConsts::Add(const OnnxConst var, std::vector<OnnxConst> &list) {
	list.push_back(var);
	std::string name = remove_chars(var.GetName());
	if ((names.end() == std::find(names.begin(), names.end(), name))) {
		names.push_back(name);
	}
	else {
		std::cout << "var " << var.GetName() << " is already added" << std::endl; // Can´t happen. ERROR
	}
}
int OnnxConsts::GetCount() const {
	return vars.size();
}
const OnnxConst& OnnxConsts::operator[](int i) const {
	return vars[i];
}
std::vector<std::string> OnnxConsts::GetVarsAsStrings() {
	std::vector<std::string> res;
	for (const OnnxConst var : vars)
	{
		res.push_back(var.GetDataTypeString());
	}
	return res;
}

OnnxConst& OnnxConsts::operator[](int i) {
	return vars[i];
}
std::vector<OnnxConst>::const_iterator OnnxConsts::begin() const {
	return vars.begin();
}
std::vector<OnnxConst>::const_iterator OnnxConsts::end() const {
	return vars.end();
}
std::vector<OnnxConst>::iterator OnnxConsts::begin() {
	return vars.begin();
}
std::vector<OnnxConst>::iterator OnnxConsts::end() {
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