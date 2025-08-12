#pragma once
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <map>
#include <any>
#include <memory>
#include <functional>
#include "OnnxVar.h"
#include <typeinfo>

#include <sstream>
namespace toCpp {
	class PredictedDim : public std::map<std::string, int>
	{
	public:
		PredictedDim() = default;
		int GetPredictedDim(const std::string& name) { return (*this)[name]; }

		int TryGetDimension(const std::string& name) {
			if (find(name) == end()) {
				(*this)[name] = -1;
				return -1; // New Dim registered
			}
			else {
				return (*this)[name];
			}
		}
		void SetDim(const std::string& name, const int dim) {
			if (find(name) == end()) {
				(*this)[name] = dim;
			}
			else {
				if ((*this)[name] == -1) {
					(*this)[name] = dim; // Set new dimension

				}
				else {
					std::cout << "Warning: Predicted dimension for " << name << " is already set to " << (*this)[name] << ", but trying to set it to " << dim << std::endl;
				}
			}
		}
		std::string Print() const {
			std::string res = "//Predicted Dimensions:\n ";
			for (const auto& dim : (*this)) {
				res += "int " + dim.first + "= " + std::to_string(dim.second) + ";\n";
			}
			return res;
		}
	};
	//forward declaration of OnnxNode
	class OnnxNode;
	// Forward declaration of OnnxGraph;
	class OnnxGraph;

	// Handler for Operators to map the Operator name to its specific Funktionality
	class OperatorHandler {
	public:
		OperatorHandler() : node(nullptr) {} // Default constructor
		OperatorHandler(const OnnxNode* node) : node(node) {};
		virtual ~OperatorHandler() = default;
		virtual bool OperatorSpecificNodeGeneration() const { return false; }
		virtual bool OperatorSpecificVarGeneration() const { return false; }
		virtual bool OperatorSpecificTensorTypes() const { return false; }
		virtual bool OperatorSpecificPreProcess() const { return false; }
		virtual bool OperatorNeedsInclude() const { return true; }
		virtual void GetOpSpecificNodeGenString(std::ostringstream& stream) const {}
		virtual void GetOpSpecificVarGen(std::ostringstream& stream) {}
		virtual void SetOpSpecificTensorTypes() {}
		virtual void PreProcess() {}
	protected:
		const OnnxNode* node;
	};


	// Onnx Node
	class OnnxNode
	{
	public:
		OnnxNode(onnx::NodeProto nodeProto, OnnxGraph* graphPtr);
		std::string GetName() const;
		std::string GetOpType() const { return op_type; }
		void GetNodeString(std::ostringstream& stream);
		void GetOpSpecificVarGen(std::ostringstream& stream);
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
		void CreateFunctionCall(std::ostringstream& stream) const;
		//void PredictDims(const OnnxVars& varsList);
		//static std::string PrintPredictedDims();#
		void SetOpSpecificTensorTypes();
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
		OperatorHandler* Handler() const { return handler.get(); }
		bool HasHandler() const { return  handler != nullptr; }
		OnnxGraph* GetGraph() const { return graph; }
	private:
		std::vector<std::string> inputNames;
		std::vector<std::string> outputNames;
		std::string name;
		std::string op_type;
		std::map<std::string, std::any> attributes; // name, value
		std::vector<OnnxTensor*> inputs;
		std::vector<OnnxTensor*> outputs;
		static PredictedDim predictedDims;
		std::unique_ptr<OperatorHandler> handler; // Operator handler for this node
		OnnxGraph* graph = nullptr; // Pointer to the graph this node belongs to, if any
	};

	class OnnxNodes
	{
	public:
		~OnnxNodes() {
			Clear();
		}
		OnnxNodes() = default;
		OnnxNodes(const OnnxNodes&) = delete;
		OnnxNodes& operator=(const OnnxNodes&) = delete;
		OnnxNodes(OnnxNodes&& other) noexcept
			: nodes(std::move(other.nodes)) 
		{
			other.nodes.clear();
		}
		OnnxNodes& operator=(OnnxNodes&& other) noexcept {
			if (this != &other) {
				Clear(); 
				nodes = std::move(other.nodes); 
				other.nodes.clear(); 
			}
			return *this;
		}
		void Clear() {
			for (auto* node : nodes) {
				delete node;
			}
			nodes.clear();
		}
		void InitWithGraph(onnx::GraphProto graph, OnnxGraph* graphPtr);
		int GetCount() const;
		void Add(const OnnxNode* var);
		const OnnxNode* operator[](int i) const;
		OnnxNode* operator[](int i);
		std::vector<OnnxNode*>::const_iterator begin() const;
		std::vector<OnnxNode*>::const_iterator end() const;
		std::vector<OnnxNode*>::iterator begin();
		std::vector<OnnxNode*>::iterator end();
		// Print predicted dimensions for all nodes
	   // names
		std::vector<std::string> GetOpTypes() const;
		std::string GetOpType(const int i) const;
		int GetOpTypeCount() const;
		void RegisterVariables(OnnxVars& varsList);

	private:
		std::vector<OnnxNode*> nodes;
		static std::vector<std::string> opTypes;
		// Static variable to store predicted dimensions for all nodes
	};


	class OperatorHandlerFactory {
	public:
		using Creator = std::function<std::unique_ptr<OperatorHandler>(const OnnxNode*)>;
		using Map = std::map<std::string, Creator>; // OpType, OperatorHandler

		static bool registerHandler(const std::string& name, Creator creator) {
			//std::cout << name << std::endl; 
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
} // namespace toCpp
// Makro für die Registrierung  
#define REGISTER_OPERATOR_HANDLER(CLASS, NAME) \
namespace { \
	static bool CLASS##_registered = OperatorHandlerFactory::registerHandler(NAME, [](const OnnxNode* node) { \
		return std::make_unique<CLASS>(node); \
    }); \
}
