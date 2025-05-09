#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
class OnnxNode
{
public:
	OnnxNode(onnx::NodeProto nodeProto);
	std::string GetName() const;
	std::string GetOpType() const { return op_type; }
	std::string GetVarInitString() const;
private:
	std::vector<std::string> input;
	std::vector<std::string> output;
	std::string name;
	std::string op_type;	
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


