#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
class OnnxVar
{
public:
	OnnxVar(onnx::ValueInfoProto valueInfo, bool isInitialising = false, bool isOutput = false);
	std::string GetName() const;
	std::string GetShapeName() const;
	onnx::TypeProto GetTypeProto() const;
	std::string GetDataTypeString() const;
	std::string GetVarInitString() const;
private:
	std::string name;
	onnx::TypeProto typeProto;
	bool isOutput = false;
	bool isInitialising = false; // true if this variable is initialised in the model
	
};

class OnnxVars
{
public:
	// Vars
	void Clear() { vars.clear(); }
	void InitWithList(const ::google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& list, bool isInitialising = false, bool isOutput = false);
	int GetCount() const;
	void Add(const OnnxVar var, std::vector<OnnxVar> &list);
	std::vector<std::string> GetVarsAsStrings();
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
	static std::vector<std::string> names;

};


