#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <variant>
#include <any>

#include "OnnxTensor.h"
namespace toCpp {
	class OnnxConst : public OnnxTensor
	{
	public:
		using OnnxTensor::Shape;
		OnnxConst(onnx::TensorProto & tensorProto);
		void Shape(::google::protobuf::RepeatedField<int64_t>);
		std::string GetShapeName() const;
		std::string GetDataAsString(bool const doInitialize);
		std::string GetConstantString(bool const doInitialize = true);
		std::vector<std::any> GetDataAsAny() const;
		size_t GetDataSize() const;
		template <typename T>
		std::vector<T>GetDataAsT() const;
		template <typename T>
		std::string GenerateNestedInitializerFromAny() const;
		using TensorData = std::variant<
			std::vector<float>,
			std::vector<std::string>,
			std::vector<bool>,
			std::vector<double>,
			std::vector<int8_t>,
			std::vector<int16_t>,
			std::vector<int32_t>,
			std::vector<int64_t>,
			std::vector<uint8_t>,
			std::vector<uint16_t>,
			std::vector<uint32_t>,
			std::vector<uint64_t>
		>;
		std::string PrintShape();
		std::string PrintReshape();
		TensorData GetData() const;
		void FillData(const onnx::TensorProto& tensorProto);
		void PrePrint();
		template<typename T>
		static std::vector<T> ExtractDataFromTensor(const onnx::TensorProto& tensorProto);
	private:
		TensorData data;
	};

	class OnnxConsts
	{
	public:
		// Vars
		void Clear() { consts.clear(); }
		void InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto>& list);
		int GetCount() const;
		void Add(const OnnxConst var);
		std::vector<std::string> GetConstsAsStrings() const;
		const OnnxConst& operator[](int i) const;
		OnnxConst& operator[](int i);
		std::deque<OnnxConst>::const_iterator begin() const;
		std::deque<OnnxConst>::const_iterator end() const;
		std::deque<OnnxConst>::iterator begin();
		std::deque<OnnxConst>::iterator end();
		bool FindConstPointerByName(const std::string name, OnnxConst*& OutputConst) const;
		// names
		std::vector<std::string> GetNames() const;
		std::string GetName(const int i) const;
		int GetNameCount() const;
	private:
		std::deque<OnnxConst> consts;
		static std::vector<std::string> names;

	};
} // namespace toCpp

