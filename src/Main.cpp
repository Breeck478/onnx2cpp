#include "onnx2cpp.h"
#include <onnx/common/file_utils.h>

#include <fstream>
#include <iostream>
using namespace toCpp;
int main(int argc, char* argv[]) {
	onnx2cpp onnx2cppInstance = onnx2cpp();
	try {
		onnx2cppInstance.ParseInputs(argc, argv);
		std::fstream file;
		file.open(onnx2cppInstance.OutputFileName(), std::ostringstream::out | std::ostringstream::trunc); // Open file for writing. Create a new file if file with this name does not exists or clear existing file
		if (file.is_open() && file.good()) {
			onnx::ModelProto model;
			onnx::LoadProtoFromPath(onnx2cppInstance.ModelFileName(), model);
			if (onnx2cppInstance.AllStatic())
				onnx2cpp::MakeCppFile(model, file, true);
			else
				onnx2cpp::MakeCppFile(model, file, onnx2cppInstance.StaticInputs());
			file.close();
		}
		else {
			std::cout << "Error: Could not open file for writing: " << onnx2cppInstance.OutputFileName() << std::endl;
			return 1; // Exit with error code
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1; // Exit with error code
	}
	std::cout << "C++ file generated successfully: " << onnx2cppInstance.OutputFileName() << std::endl;
	return 0;
}