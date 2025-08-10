#pragma once
#include <onnx/onnx_pb.h>
#include <string>

#include <vector>
 

namespace toCpp {
	std::string GetDataTypeString(const int enumValue);
	std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
	std::string remove_chars(const std::string& input, const std::string& chars_to_remove = "/.,: ");
	std::vector<std::string> split(const std::string& str, const std::string& delimiter);

	template<typename TOut>
	std::vector<TOut> ParseByteData(const std::string& dataField) {
		std::vector<TOut> result;

		if (dataField.size() <= 0) {
			return result; ; // throw std::runtime_error("Tensor has no raw_data field.");
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

	template<typename TIn, typename TOut>
	std::vector<TOut> ParseRepeatedField(const ::google::protobuf::RepeatedField<TIn> &rpf) {
		std::vector<TOut> result;

		if (rpf.size() <= 0) {
			return result; ; // throw std::runtime_error("ERROR(ParseRepeatedField): Given repeated field does not hold any Data");
		}

		size_t count = rpf.size();
		if constexpr (!std::is_same_v<TOut, bool>) {
			if (count >= sizeof(TOut)) {
				count = count / sizeof(TOut);
			}

			result.resize(count);
			if constexpr (!std::is_same_v<TOut, std::string>) {
				for (int i = 1; i < rpf.size(); ++i) {
					result.push_back(static_cast<TOut>(rpf.Get(i)));
				}
			}else{
				for (int i = 1; i < rpf.size(); ++i) {
					result.push_back(std::to_string(rpf.Get(i)));
				}
			}
		}



		return result;
	}

	template<typename TIn, typename TOut>
	std::vector<TOut> ParseRepeatedField(const ::google::protobuf::RepeatedPtrField<TIn> &rpf) {
		std::vector<TOut> result;

		if (rpf.size() <= 0) {
			return result; ; // throw std::runtime_error("ERROR(ParseRepeatedFiel): Given repeated field does not hold any Data");
		}

		size_t count = static_cast<int64_t>(rpf.size());
		if constexpr (!std::is_same_v<TOut, bool>) {
			if (count >= sizeof(TOut)) {
				count = count / sizeof(TOut);
			}

			result.resize(count);
			if constexpr (!std::is_same_v<TOut, std::string>) {
				if constexpr (std::is_same_v<TIn, std::string>) {
					std::string rpf_str = "";
					for (const auto& str : rpf)
						rpf_str += str;
					return ParseByteData<TOut>(rpf_str);
				}
				else { // need else because compiler does not register that TIn is not a string
					for (int i = 1; i < rpf.size(); ++i) {
						TIn tmp = rpf.Get(i);
						result.push_back(static_cast<TOut>(tmp));
					}
				}
			}
			else {
				for (int i = 1; i < rpf.size(); ++i) {
					result.push_back(rpf.Get(i));
				}
			}
		}

		return result;
	}
	template<typename TIn, typename TOut>
	std::vector<TOut> ParseRepeatedFieldBool(const ::google::protobuf::RepeatedField<TIn>& rpf) {
		std::vector<TOut> result;

		if (rpf.size() <= 0) {
			return result; ; // throw std::runtime_error("ERROR(ParseRepeatedField): Given repeated field does not hold any Data");
		}

		size_t count = rpf.size();
		if constexpr (!std::is_same_v<TOut, bool>) {
			throw std::runtime_error("ERROR(ParseRepeatedFieldBool): Given Datatype is not bool");
		}
		// Spezielle Behandlung für bool (ein Byte pro bool)
		count = rpf.size(); // Annehmen: jedes Byte entspricht einem bool
		result.resize(count);
		for (size_t i = 0; i < count; ++i) {
			result[i] = rpf[i] != 0;
		}

		return result;
	}

} // namespace toCpp