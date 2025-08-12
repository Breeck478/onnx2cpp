#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "OnnxGraph.h"
#include <onnx/onnx_pb.h>
namespace toCpp {
	class onnx2cpp {
	public:
		static std::string MakeCppFile(onnx::ModelProto& model, std::ostream& stream, int batchSize = 1, std::vector<std::string> staticInputs = std::vector<std::string>(), std::vector<std::string> staticOutputs = std::vector<std::string>());
		static OnnxGraph MakeCppFileGraphOut(onnx::ModelProto& model, std::ostream& stream, std::vector<std::string> staticInputs, std::vector<std::string> staticOutputs);
		static std::string MakeCppFile(onnx::ModelProto& model, std::ostream& stream, bool allStatic);
		void ParseInputs(int argc, char* argv[]);
		std::string ModelFileName() { return modelFileName; }
		std::string OutputFileName() { return outputFileName; }
		int BatchSize() { return batchSize; }
		std::vector<std::string> StaticInputs() { return staticInputs; }
		std::vector<std::string> StaticOutputs() { return staticOutputs; }
	private:
		std::string modelFileName = "";
		std::string outputFileName = "Model.h";
		int batchSize = 1; // Default batch size, can be changed by user input
		std::vector<std::string> staticInputs;
		std::vector<std::string> staticOutputs;
	};

}