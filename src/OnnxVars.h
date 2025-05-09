#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
class OnnxVar
{
public:
	OnnxVar(onnx::ValueInfoProto valueInfo);
	std::string GetName() const;
	onnx::TypeProto GetTypeProto() const;
	std::string GetDataTypeString() const;
	std::string GetVarInitString() const;
private:
	std::string name;
	onnx::TypeProto typeProto;
	
};

class OnnxVars
{
public:
	// Vars
	void Clear() { vars.clear(); names.clear(); }
	void InitWithGraph(onnx::GraphProto graph);	
	int GetCount() const;
	void Add(const OnnxVar var, std::vector<OnnxVar> &list);
	std::vector<OnnxVar> GetInput() const { return input; }
	std::vector<OnnxVar> GetOutput() const { return output; }
	static std::vector<std::string> GetVarsAsStrings(const std::vector<OnnxVar> list);
	const OnnxVar& operator[](int i) const;
	OnnxVar& operator[](int i);
	std::vector<OnnxVar>::const_iterator begin() const;
	std::vector<OnnxVar>::const_iterator end() const;
	std::vector<OnnxVar>::iterator begin();
	std::vector<OnnxVar>::iterator end();
	// names
	std::vector<std::string> GetNames() const;
	std::string GetName(const int i) const;
	int GetNameCount() const;
private:
	std::vector<OnnxVar> vars;
	std::vector<OnnxVar> input;
	std::vector<OnnxVar> output;
	std::vector<std::string> names;
};


