#pragma once

#include <onnx/onnx_pb.h>
#include "OnnxTensor.h"
#include "OnnxVars.h"
#include "OnnxConsts.h"
#include "OnnxNodes.h"

#include <vector>
enum GraphPosition {
	Begin,
	End
};

class OnnxGraph {
public:
	OnnxGraph(onnx::GraphProto& graph, bool isInitial = false, std::vector<std::string> staticInputs = std::vector<std::string>(), std::vector<std::string> staticOutputs = std::vector<std::string>(), bool doUseTemplate = false);
	void IsInitialGraph(bool isInitial) { this->isInitialGraph = isInitial; }
	void RegisterIOs();
	std::vector<std::string> GetInputNames() const;
	std::vector<std::string> GetOutputNames() const;
	std::vector<OnnxVar*> GetInputs() const { return vars.GetInputVars(); }
	std::vector<OnnxVar*> GetOutputs() const { return vars.GetOutputVars(); }
	std::string PrintGraph() const;
	virtual std::string PrintSpecificGraph(const GraphPosition position);
	std::string GetIncludes();
	std::string Name() const { return name; }
	void SetStaticIOs(std::vector <std::string> &inputs, std::vector<std::string> &outputs);
	void PreProcess();
	OnnxVars& GetVars() { return vars; }
	void AddExternVars(const OnnxVars& vars); // Add Vars from another source. E.g. for Loop-Operator from outside Graph
private:
	std::string name;
	OnnxVars vars;
	OnnxConsts consts;
	OnnxNodes nodes;
	bool isInitialGraph = false; 
	bool doUseTemplate = false; // only for the initial graph
	std::vector<std::string> staticInputs;
	std::vector<std::string> staticOutputs;
};