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
	return remove_chars(name);
}
std::string OnnxConst::GetShapeName() const {
	return GetName() + "_shape"; // For xtensor shape
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

size_t OnnxConst::GetDataSize() const {
	return std::visit([](auto&& arg) -> size_t {
		return arg.size();
		}, data);
}

template <typename T>
std::vector<T> OnnxConst::GetDataAsT() const {
	if (!std::holds_alternative<std::vector<T>>(data)) {
		std::cout << "ERROR: Tensor data type not supported for GetDataAsT" << std::endl;
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

std::string OnnxConst::GetDataTypeString() const {
	std::string res = "";
	//res = Utils::GetDataTypeString(tensorType.elem_type());
	//res = "";// "std::vector<T> const& " + name;
	//for (size_t i = 0; i < dims.size(); ++i)
	// 
	//{
	if (dims.size() > 0) {
		std::vector<int64_t> shape(dims.begin(), dims.end());
		res += "typename xt::xarray<T>::shape_type " + GetShapeName() + " = {";
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

	//res += "> ";
	//}
	//for (size_t i = 0; i < dims.size(); ++i)
	//{
	//	res += ">";
	//}

	/*res += "(";
	int i = 0;
	for (const auto dim : dims)
	{
		res += std::to_string(dim);
		if (i < dims.size() - 1)
			res += ", ";
		i++;
	}
	res += ") << ";
	i = 0;
	for (auto val : GetDataAsAny()) {
		if (std::any_cast<std::string>(&val) != nullptr) {
			res += std::any_cast<std::string>(val);
		}
		else if (std::any_cast<int32_t>(&val) != nullptr){
			res += std::to_string(std::any_cast<int32_t>(val));
		}
		else if (std::any_cast<int64_t>(&val) != nullptr) {
			res += std::to_string(std::any_cast<int64_t>(val));
		}
		else if (std::any_cast<double>(&val) != nullptr) {
			res += std::to_string(std::any_cast<double>(val));
		}
		else if (std::any_cast<float>(&val) != nullptr) {
			res += std::to_string(std::any_cast<float>(val));
		}
		else if (std::any_cast<uint64_t>(&val) != nullptr) {
			res += std::to_string(std::any_cast<uint64_t>(val));
		}
		else {
			std::cout << "ERROR: Tensor data type not supported" << std::endl;
		}


		if (i < GetDataAsAny().size() - 1)
			res += ", ";
		i++;
	}*/

	//res += " = ";
	if (GetDataSize() > 0) {

		std::ostringstream oss;

		oss << "xt::xarray<T> " << GetName() << " = ";

		if (std::holds_alternative<std::vector<float>>(data)) {
			oss << GenerateNestedInitializerFromAny<float>();

		}
		else if (std::holds_alternative<std::vector<int32_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int32_t>();

		}
		else if (std::holds_alternative<std::vector<int64_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<int64_t>();

		}
		else if (std::holds_alternative<std::vector<double>>(data)) {
			oss << GenerateNestedInitializerFromAny<double>();

		}
		else if (std::holds_alternative<std::vector<std::string>>(data)) {
			oss << GenerateNestedInitializerFromAny<std::string>();
			GetDataAsT<std::string>();
		}
		else if (std::holds_alternative<std::vector<uint64_t>>(data)) {
			oss << GenerateNestedInitializerFromAny<uint64_t>();

		}
		else {
			std::cout << "ERROR: Tensor data type not supported" << std::endl;
		}
		res += oss.str() + ";";
	}

	if (dims.size() > 0) {
		res += "\n" + GetName() + " = " + GetName() + ".reshape(" + GetShapeName() + ");";
	}
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