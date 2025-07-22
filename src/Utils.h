#pragma once
#include <onnx/onnx_pb.h>
#include <string>

#include <vector>
 


std::string GetDataTypeString(const int enumValue);
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
std::string remove_chars(const std::string& input, const std::string& chars_to_remove = "/.,: ");
std::vector<std::string> split(const std::string& str, const std::string& delimiter);

template<typename T>
std::vector<T> ParseByteData(const std::string& dataField) {
    std::vector<T> result;

    if (dataField.size() <= 0) {
        throw std::runtime_error("Tensor has no raw_data field.");
    }

    size_t count = dataField.size();
    if (count >= sizeof(T)) {
        count = count / sizeof(T);
    }
        
    result.resize(count);

    std::memcpy(result.data(), dataField.data(), dataField.size());

    return result;
}

template<typename T>
std::vector<T> ParseRepeatedField(const ::google::protobuf::RepeatedField<T> rpf) {
    std::vector<T> result;

    if (rpf.size() <= 0) {
        throw std::runtime_error("ERROR(ParseRepeatedField): Given repeated field does not hold any Data");
    }

    size_t count = rpf.size();
    if (count >= sizeof(T)) {
        count = count / sizeof(T);
    }
    result.resize(count);

    std::memcpy(result.data(), rpf.data(), rpf.size());

    return result;
}

template<typename T>
std::vector<T> ParseRepeatedField(const ::google::protobuf::RepeatedPtrField<T> rpf) {
    std::vector<T> result;

    if (rpf.size() <= 0) {
        throw std::runtime_error("ERROR(ParseRepeatedFiel): Given repeated field does not hold any Data");
    }

    size_t count = rpf.size();
    if (count >= sizeof(T)) {
        count = count / sizeof(T);
    }
    result.resize(count);

    std::memcpy(result.data(), rpf.data(), rpf.size());

    return result;
}