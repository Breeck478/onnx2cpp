#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <deque>
#include <vector>

#include "OnnxTensor.h"


namespace toCpp {
	class OnnxVar : public OnnxTensor
	{
	public:
		OnnxVar(onnx::ValueInfoProto valueInfo, bool isInput = false, bool isOutput = false);
		using OnnxTensor::Shape;
		void Shape(onnx::TensorShapeProto shapeProto);
		std::string GetShapeName() const;
		//onnx::TypeProto GetTypeProto() const;
		//std::string GetDataTypeString() const;
		std::string GetVariableString();
		bool ContainsUnkownDim() const { return containsUnknowDim; }
		void SetContainsUnkownDim() { containsUnknowDim = true; }
		bool IsIO() const { return isInput || isOutput; }
		bool IsInput() const { return isInput; }
		bool IsOutput() const { return isOutput; }
		void PreProcess();
		bool NeedsInit() const { return needsInit; }
		void NeedsInit(bool needsInit) { this->needsInit = needsInit; }
	private:
		bool isInput = false;
		bool isOutput = false;
		bool needsInit = true;
		bool containsUnknowDim = false; // If false, the variable is initialized by an operator
	};

	class OnnxVars
	{
	public:
		// Vars
		void Clear() { vars.clear(); }
		void AddFromList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list, bool isInput = false, bool isOutput = false);
		int GetCount() const;
		void Add(const OnnxVar var);
		std::vector<std::string> GetVarsAsStrings() const;
		std::vector<std::string> GetIOsAsStrings() const;
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
		bool FindVarPointerByName(const std::string name, OnnxVar*& OutputVar) const;
		std::vector<OnnxVar*> GetInputVars() const;
		std::vector<OnnxVar*> GetOutputVars() const;
	private:
		std::deque<OnnxVar> vars;
		std::vector<std::string> names;
		bool isIO = false; // true if this variable is an input or output variable of a Graph
	};

} // namespace toCpp
