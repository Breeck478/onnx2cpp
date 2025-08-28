#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <variant>
#include <any>
#include "Utils.h"
namespace toCpp {
	class OnnxTensor {
	public:
		virtual ~OnnxTensor() = default; // Virtual destructor for proper cleanup
		void Name(std::string name) { this->name = GetValidCName(name); }
		std::string Name() const { return GetValidCName(name); }
		void DataType(int32_t type) { this->dataType = type; }
		int32_t DataType()const { return dataType; }
		void HasStaticType(bool canStaticType) { this->hasStaticType = canStaticType; }
		bool HasStaticType() const { return hasStaticType; }
		bool operator==(const OnnxTensor& other) const {
			return (name == other.name) && (shape == other.shape) && (dataType == other.dataType) && (hasStaticType == other.hasStaticType);
		}
		bool operator!=(const OnnxTensor& other) const {
			return !(*this == other);
		}

		std::vector<int> Shape() const;
		std::string GetDataTypeAsString(const bool ignoreDynamic = false) const;
	protected:
		std::string name;
		std::vector<int> shape;
		int32_t dataType = 0;
		bool hasStaticType = true;  // Does not have type T but static type like int, float ...
									// Initial value is true because wihtout a Graph it can not be dynmic
									// The Graph sets this value to True if the name is not given in the staticInputs-List
	};

} // namespace toCpp

