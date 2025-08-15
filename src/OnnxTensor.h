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
		std::vector<int> Shape() const;
		std::string GetDataTypeAsString(const bool ignorStatic = false) const;
		void DataType(int32_t type) { this->dataType = type; }
		int32_t DataType()const { return dataType; }
		std::string GetVariableString();
		void HasStaticType(bool canStaticType) { this->hasStaticType = canStaticType; }
		bool HasStaticType() const { return hasStaticType; }
	protected:
		std::string name;
		std::vector<int> shape;
		int32_t dataType = 0;
		bool hasStaticType = true; // Does not have type T but static type like int, float ...
	};

} // namespace toCpp

