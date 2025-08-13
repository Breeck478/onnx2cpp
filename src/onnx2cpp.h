#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "OnnxGraph.h"
#include <onnx/onnx_pb.h>
namespace toCpp {
	class onnx2cpp {
	public:
		static std::string MakeCppFile(onnx::ModelProto& model, std::ostream& stream, std::vector<std::string> staticInputs = std::vector<std::string>());
		static OnnxGraph MakeCppFileGraphOut(onnx::ModelProto& model, std::ostream& stream, std::vector<std::string> staticInputs);
		static std::string MakeCppFile(onnx::ModelProto& model, std::ostream& stream, bool allStatic);
		void ParseInputs(int argc, char* argv[]);
		std::string ModelFileName() { return modelFileName; }
		std::string OutputFileName() { return outputFileName; }
		std::vector<std::string> StaticInputs() { return staticInputs; }
		bool AllStatic() { return allStatic; }
	private:
		std::string modelFileName = "";
		std::string outputFileName = "";
		bool allStatic = false;
		std::vector<std::string> staticInputs;
	};

}