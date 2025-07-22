#pragma once
#include "OnnxNodes.h"
#include "Utils.h"
#include <algorithm>
#include <variant>
#include <type_traits>

#include "OnnxGraph.h"

PredictedDim OnnxNode::predictedDims;
std::vector<std::string> OnnxNodes::opTypes;

OnnxNode::OnnxNode(onnx::NodeProto nodeProto)
{
	for (std::string val : nodeProto.input())
	{
		this->inputNames.push_back(remove_chars(val));
	}
	for (std::string val : nodeProto.output())
	{
		this->outputNames.push_back(remove_chars(val));
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
				res += "." + key + OnnxGraph(std::any_cast<onnx::GraphProto>(value)).PrintGraph();
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
std::string OnnxNode::CreateFunctionCall() const {
	std::string res = op_type + "(";
	res += join(inputNames, ", ");
	if (inputNames.size() > 0 && outputNames.size() > 0)
		res += ", ";
	res += join(outputNames, ", ");
	if (attributes.size() > 0) {
		res += ", " + GetParamsString();
	}
	res += "); // " + name;
	return res;
}

bool OnnxNode::NeedsInclude() const {
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(this);
	return !handler || handler->OperatorNeedsInclude();
}

std::string OnnxNode::GetNodeString() {
	std::string res = "";
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(this);
	if (handler) {
		if (handler->OperatorSpecificVarGeneration()) {
			res += handler->GetVarInitialisation();
		}
		else {
			//for (OnnxTensor* var : inputs) {				
			//	res += var->GetVariableString() + "\n";
			//}
		}
		if (handler->OperatorSpecificNodeGeneration()) {
			res += handler->GetNodeHandlerString();
		}
		else {
			res += CreateFunctionCall();
		}
	}
	else {
		//for (OnnxVar* var : inputs) {
			//if (var && var->ContainsUnkownDim()) {
		//		res += var->GetVariableString() + "\n";
			//}
		//}
		res += CreateFunctionCall();
	}
	return res;
}

std::string OnnxNode::GetVarInitialisation() {
	std::string res = "";
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(this);
	if (handler && handler->OperatorSpecificVarGeneration()) {
		res = handler->GetVarInitialisation();
	}
	else {
		for (OnnxTensor* var : inputs) {
			res += var->GetVariableString() + "\n";
		}
	}

	return res;
}

void OnnxNode::SetVarFromList(const OnnxVars& varsList) {
	bool res = false;
	std::vector<std::string> varNames = inputNames;
	// Search for own var in given List 
	for (auto name : varNames) {
		OnnxVar* varPointer = nullptr;
		if (varsList.FindVarPointerByName(name, varPointer) ) { // && !varPointer->ContainsUnkownDim()
 			inputs.push_back(varPointer);
		}
	}
	varNames = outputNames;
	for (auto name : varNames) {
		OnnxVar* outputVar = nullptr;
		if (varsList.FindVarPointerByName(name, outputVar) ) { // && !outputVar->ContainsUnkownDim()
			outputs.push_back(outputVar);
		}
	}
	SetTensorTypes();
}

void OnnxNode::SetTensorTypes() {
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(this);
	if (handler && handler->OperatorSpecificTensorTypes()) {
		handler->SetTensorTypes();
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

void OnnxNode::PreProcess() {
	std::unique_ptr<OperatorHandler> handler = OperatorHandlerFactory::create(this);
	if (handler && handler->OperatorSpecificPreProcess()) {
		handler->PreProcess();
		return;
	}
}
	
//void OnnxNode::PredictDims(const OnnxVars& varsList) {
//	//if (outputs.size() == 1 && outputs[0]->ContainsUnkownDim()) {
//
//	//}
//	for (OnnxVar* var : inputs) {
//		if (var->ContainsUnkownDim()) {
//			const auto& dims = var->GetTypeProto().tensor_type().shape().dim();
//			for (const auto dim : dims) {
//				if (dim.has_dim_param() && dim.dim_param() != "batch_size") {
//					predictedDims.SetDim(dim.dim_param(), -1);
//				}
//			}
//		}
//	}
//	for (OnnxVar* var : outputs) {
//		if (var->ContainsUnkownDim()) {
//			const onnx::TypeProto_Tensor tensorType = var->GetTypeProto().tensor_type();
//			const auto& dims = tensorType.shape().dim();
//			for (const auto dim : dims) {
//				if (dim.has_dim_param() && dim.dim_param() != "batch_size") {
//					std::string name = dim.dim_param();
//					predictedDims.SetDim(name, -1);
//				}
//			}
//		}
//	}
//}

//std::string OnnxNode::PrintPredictedDims() {
//	return predictedDims.Print();
//}



// Vars
void OnnxNodes::InitWithGraph(onnx::GraphProto graph) {
	Clear();
	for (onnx::NodeProto nodeProto : graph.node()) {
		Add(OnnxNode(nodeProto));
	}
}
void OnnxNodes::Add(const OnnxNode var) {
	nodes.push_back(var);
	if (var.NeedsInclude()) {
		std::string opType = var.GetOpType();
		if ((opTypes.end() == std::find(opTypes.begin(), opTypes.end(), opType))) {
			opTypes.push_back(opType);
		}
	}
}
int OnnxNodes::GetCount() const {
	return nodes.size();
}
const OnnxNode& OnnxNodes::operator[](int i) const {
	return nodes[i];
}

OnnxNode& OnnxNodes::operator[](int i) {
	return nodes[i];
}
std::vector<OnnxNode>::const_iterator OnnxNodes::begin() const {
	return nodes.begin();
}
std::vector<OnnxNode>::const_iterator OnnxNodes::end() const {
	return nodes.end();
}
std::vector<OnnxNode>::iterator OnnxNodes::begin() {
	return nodes.begin();
}
std::vector<OnnxNode>::iterator OnnxNodes::end() {
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

void OnnxNodes::RegisterVariables(OnnxVars& varsList) {
	for (OnnxNode& node : nodes) {
		node.SetVarFromList(varsList);

	}
}

