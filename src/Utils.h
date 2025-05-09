#include <onnx/onnx_pb.h>
#include <string>
 
class Utils {
public:
	static std::string GetDataTypeString(const int enumValue);
};
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
