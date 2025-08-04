#pragma once
#include <onnx/onnx_pb.h>
#include <string>

#include <vector>
 


std::string GetDataTypeString(const int enumValue);
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
std::string remove_chars(const std::string& input, const std::string& chars_to_remove = "/.,: ");
std::vector<std::string> split(const std::string& str, const std::string& delimiter);

template<typename TOut>
std::vector<TOut> ParseByteData(const std::string& dataField) {
    std::vector<TOut> result;

    if (dataField.size() <= 0) {
        throw std::runtime_error("Tensor has no raw_data field.");
    }

    size_t count = dataField.size();
    if constexpr (!std::is_same_v<TOut, bool>) {
        if (count >= sizeof(TOut)) {
            count = count / sizeof(TOut);
        }

        result.resize(count);
        std::memcpy(result.data(), dataField.data(), dataField.size());
    }
    else {
        // Spezielle Behandlung für bool (ein Byte pro bool)
        count = dataField.size(); // Annehmen: jedes Byte entspricht einem bool
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = dataField[i] != 0;
        }
    }

    return result;
}

template<typename TIn,typename TOut>
std::vector<TOut> ParseRepeatedField(const ::google::protobuf::RepeatedField<TIn> rpf) {
    std::vector<TOut> result;

    if (rpf.size() <= 0) {
        throw std::runtime_error("ERROR(ParseRepeatedField): Given repeated field does not hold any Data");
    }

    size_t count = rpf.size();
    if constexpr (!std::is_same_v<TOut, bool>) {
        if (count >= sizeof(TOut)) {
            count = count / sizeof(TOut);
        }

        result.resize(count);
        std::memcpy(result.data(), rpf.data(), rpf.size());
    }
    else {
        // Spezielle Behandlung für bool (ein Byte pro bool)
        count = rpf.size(); // Annehmen: jedes Byte entspricht einem bool
        result.resize(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = rpf[i] != 0;
        }
    }

    return result;
}

template<typename TIn, typename TOut>
std::vector<TOut> ParseRepeatedField(const ::google::protobuf::RepeatedPtrField<TIn> rpf) {
    std::vector<TOut> result;

    if (rpf.size() <= 0) {
        throw std::runtime_error("ERROR(ParseRepeatedFiel): Given repeated field does not hold any Data");
    }

    size_t count = rpf.size();
    if (count >= sizeof(TOut)) {
        count = count / sizeof(TOut);
    }
    result.resize(count);

    std::memcpy(result.data(), rpf.data(), rpf.size());

    return result;
}