#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <variant>
#include <any>
class OnnxConst
{
public:
	OnnxConst(onnx::TensorProto valueInfo);
	std::string GetName() const;
	void SetName(std::string name) { this->name = name; }
	std::string GetShapeName() const;
	const ::google::protobuf::RepeatedField<int64_t> GetDims() const;
	std::string GetDataTypeString() const;
	std::string GetConstantString() const;
	std::vector<std::any> GetDataAsAny() const;
	size_t GetDataSize() const;
	template <typename T>
	std::vector<T>GetDataAsT() const;
	template <typename T>
	std::string GenerateNestedInitializerFromAny() const;
	using TensorData = std::variant<
		std::vector<float>,
		std::vector<int32_t>,
		std::vector<int64_t>,
		std::vector<double>,
		std::vector<std::string>,
		std::vector<uint64_t>
	>;
	TensorData GetData() const;
private:
	std::string name;
	::google::protobuf::RepeatedField<int64_t> dims;
	TensorData data; 
};

class OnnxConsts
{
public:
	// Vars
	void Clear() { vars.clear(); }
	void InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::TensorProto>& list);
	int GetCount() const;
	void Add(const OnnxConst var);
	std::vector<std::string> GetVarsAsStrings();
	const OnnxConst& operator[](int i) const;
	OnnxConst& operator[](int i);
	std::deque<OnnxConst>::const_iterator begin() const;
	std::deque<OnnxConst>::const_iterator end() const;
	std::deque<OnnxConst>::iterator begin();
	std::deque<OnnxConst>::iterator end();
	// names
	std::vector<std::string> GetNames() const;
	std::string GetName(const int i) const;
	int GetNameCount() const;
private:
	std::deque<OnnxConst> vars;
	static std::vector<std::string> names;

};


