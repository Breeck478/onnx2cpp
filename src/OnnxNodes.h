#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <map>
#include <any>
#include <memory>
#include <functional>
#include "OnnxVars.h"

class PredictedDim
{
public:
	PredictedDim() = default;
	int GetPredictedDim(const std::string& name) { return dims[name]; }
	int TryGetDimension(const std::string& name) {
		if (dims.find(name) == dims.end()) {
			dims[name] = -1;
			return -1; // New Dim registered
		}
		else {
			return dims[name];
		}
	}
	void SetDim(const std::string& name, const int dim) {
		if (dims.find(name) == dims.end()) {
			dims[name] = dim;
		}
		else {
			if (dims[name] == -1) {
				dims[name] = dim; // Set new dimension

			}
			else {
				std::cout << "Warning: Predicted dimension for " << name << " is already set to " << dims[name] << ", but trying to set it to " << dim << std::endl;
			}
		}
	}
	std::string Print() const {
		std::string res = "//Predicted Dimensions:\n ";
		for (const auto& dim : dims) {
			res += "int " + dim.first + "= " + std::to_string(dim.second) + ";\n";
		}
		return res;
	}
private:
	std::map<std::string, int> dims;
};
class OnnxNode
{
public:
	OnnxNode(onnx::NodeProto nodeProto);
	std::string GetName() const;
	std::string GetOpType() const { return op_type; }
	std::string GetNodeString();
	std::string GetVarInitialisation();
	std::string GetParamsString()const;
	std::vector<std::string> GetInputNames() const { return inputNames; }
	std::vector<std::string> GetOutputNames() const { return outputNames; }
	std::vector<OnnxTensor*> GetInputs() const { return inputs; }
	std::vector<OnnxTensor*> GetOutputs() const { return outputs; }
	std::map<std::string, std::any> GetAttributes() const { return attributes; }
	std::any GetAttribute(const std::string& name) const {
		auto it = attributes.find(name);
		if (it != attributes.end()) {
			return it->second;
		}
		return attributes.end();
	}
	void SetVarFromList(const OnnxVars& var);
	std::string CreateFunctionCall() const;
	//void PredictDims(const OnnxVars& varsList);
	//static std::string PrintPredictedDims();#
	void SetTensorTypes();
	bool NeedsInclude() const;
	OnnxTensor* FindTensorByName(const std::string& name) const {
		for (OnnxTensor* tensor : inputs) {
			if (tensor->Name() == name) {
				return tensor;
			}
		}
		for (OnnxTensor* tensor : outputs) {
			if (tensor->Name() == name) {
				return tensor;
			}
		}
		return nullptr; // Not found
	}
	void PreProcess();
private:
	std::vector<std::string> inputNames;
	std::vector<std::string> outputNames;
	std::string name;
	std::string op_type;	
	std::map<std::string, std::any> attributes; // name, value
	std::vector<OnnxTensor*> inputs;
	std::vector<OnnxTensor*> outputs;
	static PredictedDim predictedDims;
};

class OnnxNodes
{
public:
	// Vars
	void Clear() { nodes.clear(); }
	void InitWithGraph(onnx::GraphProto graph);	
	int GetCount() const;
	void Add(const OnnxNode var);
	const OnnxNode& operator[](int i) const;
	OnnxNode& operator[](int i);
	std::vector<OnnxNode>::const_iterator begin() const;
	std::vector<OnnxNode>::const_iterator end() const;
	std::vector<OnnxNode>::iterator begin();
	std::vector<OnnxNode>::iterator end();
	 // Print predicted dimensions for all nodes
	// names
	std::vector<std::string> GetOpTypes() const;
	std::string GetOpType(const int i) const;
	int GetOpTypeCount() const;
	void RegisterVariables(OnnxVars& varsList);
	
private:
	std::vector<OnnxNode> nodes;
	static std::vector<std::string> opTypes;
	 // Static variable to store predicted dimensions for all nodes
};

// HAndler for Operators to map the Operator name to its specific Funktionality
class OperatorHandler {
public:
	OperatorHandler(const OnnxNode* node) : node(node) {};
	virtual ~OperatorHandler() = default;
	virtual bool OperatorSpecificNodeGeneration() const { return false; }
	virtual bool OperatorSpecificVarGeneration() const { return false; }
	virtual bool OperatorSpecificTensorTypes() const { return false; }
	virtual bool OperatorSpecificPreProcess() const { return false; }
	virtual bool OperatorNeedsInclude() const { return true; }
	virtual std::string GetNodeHandlerString() const { return ""; }
	virtual std::string GetVarInitialisation() { return ""; }
	virtual void PreProcess() {}
	virtual void SetTensorTypes() {}
 protected:
	const OnnxNode* node;
};

class OperatorHandlerFactory {
public:
	using Creator = std::function<std::unique_ptr<OperatorHandler>(const OnnxNode*)>;
	using Map = std::map<std::string, Creator>; // OpType, OperatorHandler

	static bool registerHandler(const std::string& name, Creator creator) {
		std::cout << name << std::endl; 
		GetMap()[name] = creator;
		return true;
	}

	static std::unique_ptr<OperatorHandler> create(const OnnxNode* node) {
		auto& map = GetMap();
		auto it = map.find(node->GetOpType());
		if (it != map.end()) {
			return it->second(node);
		}
		return nullptr;
	}

private:
	static Map& GetMap() {

		static Map map;
		return map;
	}
};

// Makro für die Registrierung  
#define REGISTER_OPERATOR_HANDLER(CLASS, NAME) \
           namespace { \
               static bool CLASS##_registered = OperatorHandlerFactory::registerHandler(NAME, [](const OnnxNode* node) { \
                   return std::make_unique<CLASS>(node); \
               }); \
           }