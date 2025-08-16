#include "OnnxNode.h"
#include "Utils.h"
#include <algorithm>
#include <variant>
#include <type_traits>

#include "OnnxGraph.h"
using namespace toCpp;
std::vector<std::string> OnnxNodes::opTypes;

OnnxNode::OnnxNode(onnx::NodeProto nodeProto, OnnxGraph* graphPtr)
{
	this->graph = graphPtr;
	for (std::string val : nodeProto.input())
	{
		this->inputNames.push_back(GetValidCName(val));
	}
	for (std::string val : nodeProto.output())
	{
		this->outputNames.push_back(GetValidCName(val));
	}
	for (onnx::AttributeProto att : nodeProto.attribute())
	{
		switch (att.type())
		{
		case onnx::AttributeProto_AttributeType_FLOAT:
			attributes[att.name()] = att.f();
			break;
		case onnx::AttributeProto_AttributeType_INT:
			attributes[att.name()] = att.i();
			break;			
		case onnx::AttributeProto_AttributeType_STRING:
			attributes[att.name()] = att.s();
			break;
		case onnx::AttributeProto_AttributeType_TENSOR:
			attributes[att.name()] = att.t();
			break;
		case onnx::AttributeProto_AttributeType_GRAPH:
			attributes[att.name()] = att.g();
			break;
		case onnx::AttributeProto_AttributeType_TYPE_PROTO:
			attributes[att.name()] = att.tp();
			break;
		case onnx::AttributeProto_AttributeType_FLOATS:
			attributes[att.name()] = std::vector<float>(att.floats().begin(), att.floats().end());
			break;
		case onnx::AttributeProto_AttributeType_INTS:
			attributes[att.name()] = std::vector<int64_t>(att.ints().begin(), att.ints().end());	
			break;
		case onnx::AttributeProto_AttributeType_STRINGS:
			attributes[att.name()] = std::vector<std::string>(att.strings().begin(), att.strings().end());
			break;
		case onnx::AttributeProto_AttributeType_TENSORS:
			attributes[att.name()] = std::vector<onnx::TensorProto>(att.tensors().begin(), att.tensors().end());
			break;
		case onnx::AttributeProto_AttributeType_GRAPHS:
			attributes[att.name()] = std::vector<onnx::GraphProto>(att.graphs().begin(), att.graphs().end());
			break;
		case onnx::AttributeProto_AttributeType_SPARSE_TENSOR:
			throw std::runtime_error("ERROR(OnnxNode::OnnxNode): Sparse tensors are not supported yet for " + nodeProto.name());
			break;
		case onnx::AttributeProto_AttributeType_TYPE_PROTOS:
			attributes[att.name()] = std::vector<onnx::TypeProto>(att.type_protos().begin(), att.type_protos().end());
			break;
		case onnx::AttributeProto_AttributeType_UNDEFINED:
			std::cout << "Warning: Undefined attribute type for " << att.name() << std::endl;
			// Do nothing, undefined type
			break;
		default:
			std::cout << "Warning: Unknown attribute type for " << att.name() << att.type() << std::endl;
			break;
		}
			}
	this->name = GetValidCName(nodeProto.name());
	this->op_type = nodeProto.op_type();
	this->handler = OperatorHandlerFactory::create(this);
}

std::string OnnxNode::GetName() const{
	return name;
}

std::string OnnxNode::GetParamsString() const {
	std::string res = "";
	if (attributes.size() > 0) {
		int counter = 1;
		res += op_type + "Params{";
		for (auto it = attributes.begin(); it != attributes.end(); ++it) {
			auto& [key, value] = *it;
			if (value.type() == typeid(float)) {
				res += "." + key + "= " + std::to_string(std::any_cast<float>(value)) + "";
			}
			else if (value.type() == typeid(int64_t)) {
				res += "." + key + "= " + std::to_string(std::any_cast<int64_t>(value)) + "";
			}
			else if (value.type() == typeid(std::string)) {
				res += "." + key + "= \"" + std::any_cast<std::string>(value) + "\"";
			}
			else if (value.type() == typeid(onnx::TensorProto)) {
				res += "." + key + "= " + std::any_cast<onnx::TensorProto>(value).name() + "";
			}
			else if (value.type() == typeid(std::vector<float>)) {
				res += "." + key + ": [";
				const auto& floats = std::any_cast<std::vector<float>>(value);
				for (const auto& val : floats) {
					res += std::to_string(val) + ", ";
				}
				res += "]";
			}
			else if (value.type() == typeid(std::vector<int64_t>)) {
				res += "." + key + ": [";
				const auto& ints = std::any_cast<std::vector<int64_t>>(value);
				for (const auto& val : ints) {
					res += std::to_string(val) + ", ";
				}
				res += "]";
			}
			else if (value.type() == typeid(std::vector<std::string>)) {
				res += "." + key + ": [" + Join(std::any_cast<std::vector<std::string>>(value), ", ") + "]";
			}
			else if (value.type() == typeid(std::vector<onnx::TensorProto>)) {
				res += "." + key + ": [";
				const auto& tensors = std::any_cast<std::vector<onnx::TensorProto>>(value);
				for (const auto& tensor : tensors) {
					res += tensor.name() + ", ";
				}
				res += "]";
			}
			else if (value.type() == typeid(onnx::GraphProto)) {
				std::cout << "Warning: Graph attribute should be handled specificly for node " << GetName() << std::endl;
			}
			else if (value.type() == typeid(onnx::TypeProto)) {
				std::cout << "Warning: TypeProto attribute should be handled specificly for node " << GetName() << std::endl;
			}
			else {
				res += "." + key + "= UnknownType";
			}
			if (counter < attributes.size()) res += ", ";
			counter += 1;
		}

		res += "}";

	}
	return res;
}
void OnnxNode::CreateFunctionCall(std::ostringstream & stream) const {
	stream << op_type + "(";
	stream << Join(inputNames, ", ");
	if (inputNames.size() > 0 && outputNames.size() > 0)
		stream << ", ";
	stream << Join(outputNames, ", ");
	if (attributes.size() > 0) {
		stream << ", " + GetParamsString();
	}
	stream << "); // " + name;
}

bool OnnxNode::NeedsInclude() const {
	return !HasHandler() || Handler()->OperatorNeedsInclude();
}

void OnnxNode::GetNodeString(std::ostringstream & stream) {
	if (HasHandler()) {
		if (Handler()->OperatorSpecificNodeGeneration()) {
			Handler()->GetOpSpecificNodeGenString(stream);
		}
		else {
			CreateFunctionCall(stream);
		}
	}
	else {
		 CreateFunctionCall(stream);
	}
	stream << "\n"; // Add a newline after the function call
}


void OnnxNode::SetTensorFromLists(const OnnxVars& vars, const OnnxConsts& consts) {
	bool res = false;
	std::vector<std::string> varNames = inputNames;
	// Search for own var in given List 
	for (auto& name : varNames) {
		OnnxVar* varPointer = nullptr;
		if (vars.FindVarPointerByName(name, varPointer) ) { 
 			inputs.push_back(varPointer);
			continue;
		}
		OnnxConst* constPointer = nullptr;
		if (consts.FindConstPointerByName(name, constPointer)) {
			inputs.push_back(constPointer);
		}
	}

	// here only vars because constants cant be outputs
	varNames = outputNames;
	for (auto& name : varNames) {
		OnnxVar* outputVar = nullptr;
		if (vars.FindVarPointerByName(name, outputVar) ) {
			outputs.push_back(outputVar);
		}
	}
	SetOpSpecificTensorTypes();
}

void OnnxNode::SetOpSpecificTensorTypes() {
	if (HasHandler() && Handler()->OperatorSpecificTensorTypes()) {
		Handler()->SetOpSpecificTensorTypes();
		return;
	}
	bool containsNonStaticInput = false;
	for (OnnxTensor* var : inputs) {
		if (!var->HasStaticType()) {
			containsNonStaticInput = true;
			break;
		}
	}
	if (containsNonStaticInput) {
		for (OnnxTensor* var : outputs) {
			var->HasStaticType(false);
		}
	}
}

void OnnxNode::PrePrint() {
	if (HasHandler() && Handler()->OperatorSpecificPreProcess()) {
		Handler()->PrePrint();
		return;
	}
}



// Vars
void OnnxNodes::InitWithGraph(onnx::GraphProto graph, OnnxGraph* graphPtr) {
	Clear();
	for (onnx::NodeProto nodeProto : graph.node()) {
		Add(new OnnxNode(nodeProto, graphPtr)); // New Node ptr gets added to the list. it will never be moved unless it gets destroyed
	}
}
void OnnxNodes::Add(const OnnxNode* var) {
	nodes.push_back(const_cast<OnnxNode*>(var));
	if (var->NeedsInclude()) {
		std::string opType = var->GetOpType();
		if ((opTypes.end() == std::find(opTypes.begin(), opTypes.end(), opType))) {
			opTypes.push_back(opType);
		}
	}
}
int OnnxNodes::GetCount() const {
	return nodes.size();
}
const OnnxNode* OnnxNodes::operator[](int i) const {
	return nodes[i];
}

OnnxNode* OnnxNodes::operator[](int i) {
	return nodes[i];
}
std::vector<OnnxNode*>::const_iterator OnnxNodes::begin() const {
	return nodes.begin();
}
std::vector<OnnxNode*>::const_iterator OnnxNodes::end() const {
	return nodes.end();
}
std::vector<OnnxNode*>::iterator OnnxNodes::begin() {
	return nodes.begin();
}
std::vector<OnnxNode*>::iterator OnnxNodes::end() {
	return nodes.end();
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

void OnnxNodes::RegisterTensors(OnnxVars& varsList, OnnxConsts& constList) {
	for (OnnxNode* node : nodes) {
		node->SetTensorFromLists(varsList, constList);

	}
}

