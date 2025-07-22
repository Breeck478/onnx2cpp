#pragma once
#include "Utils.h"

std::string GetDataTypeString(const int enumValue) {
    switch (enumValue)
    {
    case    onnx::TensorProto_DataType_UNDEFINED: return "undefined";
    case    onnx::TensorProto_DataType_FLOAT: return "float";
    case    onnx::TensorProto_DataType_STRING:  return "std::string";
    case    onnx::TensorProto_DataType_BOOL: return "bool";
    case    onnx::TensorProto_DataType_DOUBLE: return "double";
    case    onnx::TensorProto_DataType_INT8: return "int8_t";
    case    onnx::TensorProto_DataType_INT16: return "int16_t";
    case    onnx::TensorProto_DataType_INT32: return "int32_t";
    case    onnx::TensorProto_DataType_INT64: return "int64_t";
    case    onnx::TensorProto_DataType_UINT8: return "uint8_t";
    case    onnx::TensorProto_DataType_UINT16: return "uint16_t";
    case    onnx::TensorProto_DataType_UINT32: return "uint32_t";
    case    onnx::TensorProto_DataType_UINT64: return "uint64_t";
        //case    onnx::TensorProto_DataType_FLOAT16: return "float16_t";   
        //case    onnx::TensorProto_DataType_COMPLEX64: return "complex64_t"; 
        //case    onnx::TensorProto_DataType_COMPLEX128: return "complex128_t"; 
        //case    onnx::TensorProto_DataType_BFLOAT16: return "bfloat16_t"; 
        //case    onnx::TensorProto_DataType_FLOAT8E4M3FN: return "Todo";  
        //case    onnx::TensorProto_DataType_FLOAT8E4M3FNUZ: return "Todo";
        //case    onnx::TensorProto_DataType_FLOAT8E5M2: return "Todo";
        //case    onnx::TensorProto_DataType_FLOAT8E5M2FNUZ: return "Todo";
        //case    onnx::TensorProto_DataType_UINT4: return "Todo";
        //case    onnx::TensorProto_DataType_INT4: return "Todo";
        //case    onnx::TensorProto_DataType_FLOAT4E2M1: return "Todo";
    default:
        throw std::runtime_error("ERROR(GetDataTypeString) : Data Type not yet Supported");
    }
}


std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
    std::string result;
    for (size_t i = 0; i < strings.size(); ++i) {
        result += strings[i];
        if (i + 1 < strings.size()) {
            result += delimiter;
        }
    }
    return result;
}

std::string remove_chars(const std::string& input, const std::string& chars_to_remove) {
    std::string result = input;
    result.erase(std::remove_if(result.begin(), result.end(),
        [&chars_to_remove](char c) {
            return chars_to_remove.find(c) != std::string::npos;
        }),
        result.end());
    return result;
}

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
	std::vector<std::string> result;
	size_t start = 0;
	size_t end = str.find(delimiter);
	while (end != std::string::npos) {
		result.push_back(str.substr(start, end - start));
		start = end + delimiter.length();
		end = str.find(delimiter, start);
	}
	result.push_back(str.substr(start, end));
	return result;
}
