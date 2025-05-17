#include "OnnxNodes.h"
#include "Utils.h"
#include <algorithm>

OnnxNode::OnnxNode(onnx::NodeProto nodeProto)
{
	for (std::string val : nodeProto.input())
	{
		this->input.push_back(remove_chars(val));
	}
	for (std::string val : nodeProto.output())
	{
		this->output.push_back(remove_chars(val));
	}
	this->name = remove_chars(nodeProto.name());
	this->op_type = nodeProto.op_type();
}

std::string OnnxNode::GetName() const{
	return name;
}

std::string OnnxNode::GetVarInitString() const { 
	std::string res = op_type +"(";
	res += join(input, ", ");
	if (input.size() > 0 && output.size() > 0)
		res += ", ";
	res += join(output, ", ");
	res += "); // " + name;
	return res;
}



// Vars
void OnnxNodes::InitWithGraph(onnx::GraphProto graph) {
	Clear();
	for (onnx::NodeProto nodeProto : graph.node()) {
		Add(OnnxNode(nodeProto));
	}
}
void OnnxNodes::Add(const OnnxNode var) {
	vars.push_back(var);
	std::string opType = var.GetOpType();
	if ((opTypes.end() == std::find(opTypes.begin(), opTypes.end(), opType))) {
		opTypes.push_back(opType);
	}
	else {
		std::cout <<"var " << var.GetName() << " is already added" << std::endl;
	}
}
int OnnxNodes::GetCount() const {
	return vars.size();
}
const OnnxNode& OnnxNodes::operator[](int i) const {
	return vars[i];
}

OnnxNode& OnnxNodes::operator[](int i) {
	return vars[i];
}
std::vector<OnnxNode>::const_iterator OnnxNodes::begin() const {
	return vars.begin();
}
std::vector<OnnxNode>::const_iterator OnnxNodes::end() const {
	return vars.end();
}
std::vector<OnnxNode>::iterator OnnxNodes::begin() {
	return vars.begin();
}
std::vector<OnnxNode>::iterator OnnxNodes::end() {
	return vars.end();
}

// opTypes

std::string OnnxNodes::GetOpType(const int i) const {
	return opTypes[i];
}
std::vector<std::string> OnnxNodes::GetOpTypes() const {
	return opTypes;
}
int OnnxNodes::GetOpTypeCount() const {
	return opTypes.size();
}