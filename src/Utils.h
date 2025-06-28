#include <onnx/onnx_pb.h>
#include <string>

#include <vector>
 
class Utils {
public:
	static std::string GetDataTypeString(const int enumValue);
};
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
std::string remove_chars(const std::string& input, const std::string& chars_to_remove = "/.,: ");

template<typename T>
std::vector<T> ParseRawData(const onnx::TensorProto& tensor) {
    std::vector<T> result;

    if (!tensor.has_raw_data()) {
        throw std::runtime_error("Tensor has no raw_data field.");
    }

    const std::string& raw = tensor.raw_data();
    size_t count = raw.size() / sizeof(T);
    result.resize(count);

    std::memcpy(result.data(), raw.data(), raw.size());

    return result;
}
