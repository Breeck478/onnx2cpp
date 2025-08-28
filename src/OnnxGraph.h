#pragma once

#include <onnx/onnx_pb.h>
#include "OnnxTensor.h"
#include "OnnxVar.h"
#include "OnnxConst.h"
#include "OnnxNode.h"

#include <sstream>

#include <vector>
namespace toCpp {

	class OnnxGraph {
	public:
		OnnxGraph(onnx::GraphProto graph, bool isInitial = false, std::vector<std::string> staticInputs = std::vector<std::string>());
		OnnxGraph();
		void IsInitialGraph(bool isInitial) { this->isInitialGraph = isInitial; }
		void RegisterIOs();
		std::vector<std::string> GetInputNames() const;
		std::vector<std::string> GetOutputNames() const;
		std::vector<OnnxVar*> GetInputs() const { return vars.GetInputVars(); }
		std::vector<OnnxVar*> GetOutputs() const { return vars.GetOutputVars(); }
		void PrintGraph(std::ostringstream& stream) const;
		void GetIncludes(std::ostringstream& stream) const;
		std::string Name() const { return name; }
		void SetStaticIOs(std::vector <std::string>& inputs);
		void PrePrint();
		OnnxVars& GetVars() { return vars; }
		OnnxConsts& GetConsts() { return consts; }
		void AddExternVars(const OnnxVars& vars); // Add Vars from another source. E.g. for Loop-Operator from outside Graph
		void AddExternConsts(const OnnxConsts& consts); // Add Vars from another source. E.g. for Loop-Operator from outside Graph
	private:
		std::string name;
		OnnxVars vars;
		OnnxConsts consts;
		OnnxNodes nodes;
		bool isInitialGraph = false;
		bool doUseTemplate = false; // only for the initial graph
		std::vector<std::string> staticInputs;
	};
} // namespace toCpp