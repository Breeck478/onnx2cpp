#include <onnx/onnx_pb.h>
#include <string>
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
	OnnxVar(onnx::ValueInfoProto valueInfo);
	std::string GetName() const;
	onnx::TypeProto GetTypeProto() const;
	std::string GetDataTypeString() const;
	std::string GetVarInitString() const;
private:
	std::string name;
	onnx::TypeProto typeProto;

};


