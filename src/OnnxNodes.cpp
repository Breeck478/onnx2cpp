#include "OnnxNodes.h"
#include "Utils.h"
#include <algorithm>
#include <variant>
#include <type_traits>




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
			// Handle sparse tensor type if needed
			attributes[att.name()] = att.sparse_tensor();
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
	this->name = remove_chars(nodeProto.name());
	this->op_type = nodeProto.op_type();
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
			/*else if (value.type() == typeid(std::vector<float>)) {
				res += "." + key + ": [" + join(std::any_cast<std::vector<float>>(value), ", ") + "]";
			}
			else if (value.type() == typeid(std::vector<int64_t>)) {
				res += "." + key + ": [" + join(std::any_cast<std::vector<int64_t>>(value), ", ") + "]";
			}
			else if (value.type() == typeid(std::vector<std::string>)) {
				res += "." + key + ": [" + join(std::any_cast<std::vector<std::string>>(value), ", ") + "]";
			}
			else if (value.type() == typeid(std::vector<onnx::TensorProto>)) {
				res += "." + key + ": [";
				const auto& tensors = std::any_cast<std::vector<onnx::TensorProto>>(value);
				for (const auto& tensor : tensors) {
					res += tensor.name() + ", ";
				}
				res += "]";
			}*/
			else if (value.type() == typeid(onnx::GraphProto)) {
				res += "." + key + "= Graph"; // Platzhalter für Graph-Darstellung
			}
			else if (value.type() == typeid(onnx::TypeProto)) {
				res += "." + key + "= TypeProto"; // Platzhalter für TypeProto-Darstellung
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

std::string OnnxNode::GetVarInitString() const {
	std::string res = "";
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(*this);
	if (handler && handler->OperatorSpecificGeneration()) {
		res = handler->GetVarInitString();
	}
	else {
		res += op_type + "(";
		res += join(input, ", ");
		if (input.size() > 0 && output.size() > 0)
			res += ", ";
		res += join(output, ", ");
		if (attributes.size() > 0) {
			res += ", " + GetParamsString();
		}
		res += "); // " + name;
	}
		
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
	//else {
	//	std::cout <<"var " << var.GetName() << " is already added" << std::endl;
	//}
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