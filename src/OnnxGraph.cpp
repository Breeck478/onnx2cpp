#pragma once

#include "OnnxGraph.h"

#include <algorithm>
using namespace toCpp;
OnnxGraph::OnnxGraph(onnx::GraphProto graph, bool isInitial, std::vector<std::string> staticInputs, std::vector<std::string> staticOutputs): name(graph.name()), isInitialGraph(isInitial), staticInputs(staticInputs), staticOutputs(staticOutputs){
	vars.AddFromList(graph.input(), true);
	vars.AddFromList(graph.output(), false, true);
	vars.AddFromList(graph.value_info());
	consts.InitWithList(graph.initializer());
	nodes.InitWithGraph(graph, this);
	if (isInitialGraph) {
		// If this is the initial graph, we need to register IOs
		// This is done to set the static or non-static type of the IOs
		// depending on user input
		RegisterIOs();
	}
}

void OnnxGraph::RegisterIOs() {
	// Set all IOs to static or non tatic type depending on user Input
	// 
	//inputs

	for (OnnxTensor* var : vars.GetInputVars()) {
		auto it = std::find(staticInputs.begin(), staticInputs.end(), var->Name());
		if (it == staticInputs.end()) {
			doUseTemplate = true;
			var->HasStaticType(false);
		}
	}
	for (OnnxTensor* var : vars.GetOutputVars()) {
		auto it = std::find(staticOutputs.begin(), staticOutputs.end(), var->Name());
		if (it == staticOutputs.end()) {
			doUseTemplate = true;
			var->HasStaticType(false);
		}
	}
	nodes.RegisterVariables(vars);
}

void OnnxGraph::SetStaticIOs(std::vector <std::string> &inputs, std::vector<std::string> &outputs) {
	this->staticInputs = inputs;
	this->staticOutputs = outputs;
	RegisterIOs();
}

std::string OnnxGraph::PrintSpecificGraph(const GraphPosition position) {
	return "";
}

void OnnxGraph::PreProcess() {
	for (auto& constant : consts) 
		constant.PreProcess();
	for (auto& var : vars)
		var.PreProcess();
	for (auto* node : nodes) 
		node->PreProcess();
}

void OnnxGraph::AddExternVars(const OnnxVars& vars) {
	// Add Vars from another source. E.g. for Loop-Operator from outside Graph
	for (OnnxVar var : vars) {
		var.NeedsInit(false); // Do not need to initialize these vars, they are already initialized in the graph
		this->vars.Add(var);
	}
}

void OnnxGraph::PrintGraph(std::ostringstream & stream) const {
	if (isInitialGraph) {
		// start with all includes neccesary for the Operators
		GetIncludes(stream);
		stream << "#include <vector>" << std::endl;
		//stream << "#include <xtensor/xarray.hpp>" << std::endl;		
		if (doUseTemplate) {
			stream << "template <typename T>\n";
		}
		stream << "void " + Name() + "(" + join(vars.GetIOsAsStrings(), ", ") + "){\n";
	}
	else {
		stream << "auto " + Name() + " = [&](" + join(vars.GetIOsAsStrings(), ", ") + "){\n";
	}
	//res += PrintSpecificGraph(GraphPosition::Begin);
	stream << "// Vars:\n";
	for (const std::string str : vars.GetVarsAsStrings())
		stream << str + "\n";
	stream << "// Consts:\n";
	for (const std::string str : consts.GetVarsAsStrings())
		stream << str + "\n";
	stream << "// Nodes:\n";
	for (auto* node : nodes) {
		try {
			node->GetNodeString(stream);
		}
		catch (const std::exception& e) {
			std::cerr << "Error generating" << node->GetName() << "operator: " << e.what() << std::endl;
		}
	}
	//res += PrintSpecificGraph(GraphPosition::End);
	if (isInitialGraph) {
		stream << "}\n";
	}
	else {
		stream << "}; // " + Name() + "\n";
	}
}

void OnnxGraph::GetIncludes(std::ostringstream & stream) const {
	for (const std::string& type : nodes.GetOpTypes()) {
		stream << "#include <Operators/" + type + ".h>\n";
	}
}


std::vector<std::string> OnnxGraph::GetInputNames() const {
	auto inputs = vars.GetInputVars();
	std::vector<std::string> inputNames;
	for (auto& input : inputs) {
		inputNames.push_back(input->Name());
	}
	return inputNames;
}

std::vector<std::string> OnnxGraph::GetOutputNames() const {
	auto outputs = vars.GetOutputVars();
	std::vector<std::string> outputNames;
	for (auto& output : outputs) {
		outputNames.push_back(output->Name());
	}
	return outputNames;
}