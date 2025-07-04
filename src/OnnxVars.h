#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <deque>
#include <vector>




class OnnxVar
{
public:
	OnnxVar(onnx::ValueInfoProto valueInfo, bool isInitialising = false, bool isOutput = false);
	std::string GetName() const;
	std::string GetShapeName() const;
	onnx::TypeProto GetTypeProto() const;
	std::string GetDataTypeString() const;
	std::string GetVariableString();
	bool ContainsUnkownDim() const { return containsUnknowDim; }
	void SetContainsUnkownDim() { containsUnknowDim = true; }
	bool SetInitialization();
private:
	std::string name;
	onnx::TypeProto typeProto;
	bool isOutput = false;
	bool is_initialized_in_model = false; // true if this variable is initialised in the model
	bool containsUnknowDim = false; // If false, the variable is initialized by an operator
	
};

class OnnxVars
{
public:
	// Vars
	void Clear() { vars.clear(); }
	void InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list, bool isInitialising = false, bool isOutput = false);
	int GetCount() const;
	void Add(const OnnxVar var);
	std::vector<std::string> GetVarsAsStrings();
	const OnnxVar& operator[](int i) const;
	OnnxVar& operator[](int i);
	std::deque<OnnxVar>::const_iterator begin() const;
	std::deque<OnnxVar>::const_iterator end() const;
	std::deque<OnnxVar>::iterator begin();
	std::deque<OnnxVar>::iterator end();
	// names
	std::vector<std::string> GetNames() const;
	std::string GetName(const int i) const;
	int GetNameCount() const;
	bool FindConstPointerByName(const std::string name, OnnxVar*& OutputVar) const;
	void SetInitializations();
private:
	std::deque<OnnxVar> vars;
	static std::vector<std::string> names;
};


