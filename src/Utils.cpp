#include "Utils.h"

std::string Utils::GetDataTypeString(const int enumValue) {
    switch (enumValue)
    {
    case    onnx::TensorProto_DataType_UNDEFINED: return "undefined";
    case    onnx::TensorProto_DataType_FLOAT: return "float";
    case    onnx::TensorProto_DataType_UINT8: return "uint8";
    case    onnx::TensorProto_DataType_INT8: return "int8";
    case    onnx::TensorProto_DataType_UINT16: return "uint16";
    case    onnx::TensorProto_DataType_INT16: return "int16";
    case    onnx::TensorProto_DataType_INT32: return "int32";
    case    onnx::TensorProto_DataType_INT64: return "int64";
    case    onnx::TensorProto_DataType_STRING:  return "string";
    case    onnx::TensorProto_DataType_BOOL: return "boolean";
    case    onnx::TensorProto_DataType_FLOAT16: return "float16";
    case    onnx::TensorProto_DataType_DOUBLE: return "double";
    case    onnx::TensorProto_DataType_UINT32: return "uint32";
    case    onnx::TensorProto_DataType_UINT64: return "uint64";
    case    onnx::TensorProto_DataType_COMPLEX64: return "complex64";
    case    onnx::TensorProto_DataType_COMPLEX128: return "complex128";
    case    onnx::TensorProto_DataType_BFLOAT16: return "bfloat16";
    case    onnx::TensorProto_DataType_FLOAT8E4M3FN: return "Todo";
    case    onnx::TensorProto_DataType_FLOAT8E4M3FNUZ: return "Todo";
    case    onnx::TensorProto_DataType_FLOAT8E5M2: return "Todo";
    case    onnx::TensorProto_DataType_FLOAT8E5M2FNUZ: return "Todo";
    case    onnx::TensorProto_DataType_UINT4: return "Todo";
    case    onnx::TensorProto_DataType_INT4: return "Todo";
    //case    onnx::TensorProto_DataType_FLOAT4E2M1: return "Todo";
    default:
        return "ERROR";
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