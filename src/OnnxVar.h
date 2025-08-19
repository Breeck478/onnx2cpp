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
		std::string GetVariableString(const bool ignorStatic = false);
		bool IsIO() const { return isInput || isOutput; }
		bool IsInput() const { return isInput; }
		bool IsOutput() const { return isOutput; }
		// for testduit to get correct initial string
		void SetIO(bool isInput, bool isOutput) {
			this->isInput = isInput;
			this->isOutput = isOutput;
		}
		bool NeedsInit() const { return needsInit; }
		void NeedsInit(bool needsInit) { this->needsInit = needsInit; }
		bool operator==(const OnnxVar& other) const {
			return OnnxTensor::operator==(other) && (isInput == other.isInput) && (isOutput == other.isOutput) && (needsInit == other.needsInit);
		}
		bool operator!=(const OnnxVar& other) const {
			return !(*this == other);
		}
	private:
		bool isInput = false;
		bool isOutput = false;
		bool needsInit = true;
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
		bool FindVarPointerByName(const std::string name, OnnxVar*& OutputVar) const;
		std::vector<OnnxVar*> GetInputVars() const;
		std::vector<OnnxVar*> GetOutputVars() const;
	private:
		std::deque<OnnxVar> vars;
	};

} // namespace toCpp
