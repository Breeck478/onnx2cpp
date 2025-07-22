#pragma once

#include "OnnxGraph.h"

#include <algorithm>

OnnxGraph::OnnxGraph(onnx::GraphProto& graph, bool isInitial, std::vector<std::string> staticInputs, std::vector<std::string> staticOutputs, bool doUseTemplate): name(graph.name()), isInitialGraph(isInitial), staticInputs(staticInputs), staticOutputs(staticOutputs), doUseTemplate(doUseTemplate){
	vars.AddFromList(graph.input(), true);
	vars.AddFromList(graph.output(), false, true);
	vars.AddFromList(graph.value_info());
	consts.InitWithList(graph.initializer());
	nodes.InitWithGraph(graph);
	if (isInitialGraph) {
		// If this is the initial graph, we need to register IOs
		// This is done to set the static or non-static type of the IOs
		// depending on user input
		// Inputs and Outputs are set to static or non-static type
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
			var->HasStaticType(false);
		}
	}
	for (OnnxTensor* var : vars.GetOutputVars()) {
		auto it = std::find(staticOutputs.begin(), staticOutputs.end(), var->Name());
		if (it == staticOutputs.end()) {
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
	for (auto& node : nodes) 
		node.PreProcess();
}

std::string OnnxGraph::PrintGraph() {
	std::string res = "// Graph:\n";
	if (isInitialGraph) {
		if (doUseTemplate) {
			res += "template <typename T>\n";
		}
		res += "void " + Name() + "(" + join(vars.GetIOsAsStrings(), ", ") + "){\n";
	}
	else {
		res += "auto " + Name() + " = [&](" + join(vars.GetIOsAsStrings(), ", ") + "){\n";
	}
	res += PrintSpecificGraph(GraphPosition::Begin);
	res += "// Vars:\n";
	for (const std::string str : vars.GetVarsAsStrings())
		res += str + "\n";
	res += "// Consts:\n";
	for (const std::string str : consts.GetVarsAsStrings())
		res += str + "\n";
	res += "// Nodes:\n";
	for (auto& node : nodes) {
		try {
			res += node.GetNodeString() + "\n";
		}
		catch (const std::exception& e) {
			std::cerr << "Error generating" << node.GetName() << "operator: " << e.what() << std::endl;
			return 0;
		}
	}
	res += PrintSpecificGraph(GraphPosition::End);
	if (isInitialGraph) {
		res += "}\n";
	}
	else {
		res += "}; // " + Name() + "\n";
	}
	return res;
}

std::string OnnxGraph::GetIncludes() {
	std::string res = "";
	for (const std::string& type : nodes.GetOpTypes()) {
		res += "#include <Operators/" + type + ".h>\n";
	}
	return res;
}


std::vector<std::string> OnnxGraph::GetInputNames() {
	auto& inputs = vars.GetInputVars();
	std::vector<std::string> inputNames;
	for (auto& input : inputs) {
		inputNames.push_back(input->Name());
	}
	return inputNames;
}

std::vector<std::string> OnnxGraph::GetOutputNames() {
	auto& outputs = vars.GetOutputVars();
	std::vector<std::string> outputNames;
	for (auto& output : outputs) {
		outputNames.push_back(output->Name());
	}
	return outputNames;
}