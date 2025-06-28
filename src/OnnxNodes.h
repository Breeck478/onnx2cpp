#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <map>
#include <any>
#include <memory>
#include <functional>
//#include "Operators/Add.h"
class OnnxNode
{
public:
	OnnxNode(onnx::NodeProto nodeProto);
	std::string GetName() const;
	std::string GetOpType() const { return op_type; }
	std::string GetVarInitString() const;
	std::string GetParamsString()const;
	std::vector<std::string> GetInputs() const { return input; }
	std::vector<std::string> GetOutputs() const { return output; }
	std::map<std::string, std::any> GetAttributes() const { return attributes; }
	std::any GetAttribute(const std::string& name) const {
		auto it = attributes.find(name);
		if (it != attributes.end()) {
			return it->second;
		}
		return std::any();
	}

private:
	std::vector<std::string> input;
	std::vector<std::string> output;
	std::string name;
	std::string op_type;	
	std::map<std::string, std::any> attributes; // name, value
};

class OnnxNodes
{
public:
	// Vars
	void Clear() { vars.clear(); opTypes.clear(); }
	void InitWithGraph(onnx::GraphProto graph);	
	int GetCount() const;
	void Add(const OnnxNode var);
	const OnnxNode& operator[](int i) const;
	OnnxNode& operator[](int i);
	std::vector<OnnxNode>::const_iterator begin() const;
	std::vector<OnnxNode>::const_iterator end() const;
	std::vector<OnnxNode>::iterator begin();
	std::vector<OnnxNode>::iterator end();
	// names
	std::vector<std::string> GetOpTypes() const;
	std::string GetOpType(const int i) const;
	int GetOpTypeCount() const;
private:
	std::vector<OnnxNode> vars;
	std::vector<std::string> opTypes;
};

class OperatorHandler {
public:
	OperatorHandler(const OnnxNode node) : node(node) {};
	virtual ~OperatorHandler() = default;
	virtual bool OperatorSpecificGeneration() const { return false; }
	virtual std::string GetVarInitString() const { return ""; }
 protected:
	const OnnxNode node;
};

class OperatorHandlerFactory {
public:
	using Creator = std::function<std::unique_ptr<OperatorHandler>(OnnxNode)>;
	using Map = std::map<std::string, Creator>;

	static bool registerHandler(const std::string& name, Creator creator) {
		std::cout << name << std::endl; 
		GetMap()[name] = creator;
		return true;
	}

	static std::unique_ptr<OperatorHandler> create(const OnnxNode node) {
		auto& map = GetMap();
		auto it = map.find(node.GetOpType());
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
               static bool CLASS##_registered = OperatorHandlerFactory::registerHandler(NAME, [](const OnnxNode& node) { \
                   return std::make_unique<CLASS>(node); \
               }); \
           }