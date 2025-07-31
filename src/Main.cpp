#include "onnx2cpp.h"
#include <onnx/common/file_utils.h>

#include <fstream>
#include <iostream>
int main(int argc, char* argv[]) {
	onnx2cpp onnx2cppInstance = onnx2cpp();
	onnx2cppInstance.ParseInputs(argc, argv);
	std::fstream file;
	file.open(onnx2cppInstance.OutputFileName(), std::ostringstream::out | std::ostringstream::trunc); // Open file for writing. Create a new file if file with this name does not exists or clear existing file
	if (file.is_open() && file.good()) {
		onnx::ModelProto model;
		onnx::LoadProtoFromPath(onnx2cppInstance.ModelFileName(), model);
		onnx2cpp::MakeCppFile(model, file, onnx2cppInstance.BatchSize(), onnx2cppInstance.StaticInputs(), onnx2cppInstance.StaticOutputs());
		file.close();
	}
	else {
		std::cout << "Error: Could not open file for writing: " << onnx2cppInstance.OutputFileName() << std::endl;
		return 1; // Exit with error code
	}
	std::cout << "C++ file generated successfully: " << onnx2cppInstance.OutputFileName() << std::endl;
	return 0;
}