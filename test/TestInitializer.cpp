#include "onnx2cpp.h"
#include "Utils.h"
#include "OnnxConst.h"
#include "OnnxVar.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <onnx/common/file_utils.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <memory>
#include <vector> // Include vector for dynamic memory allocation

bool load_input_data(const std::string& filename, onnx::TensorProto& result)
{
   /* TODO: read Protobuffers documentation. This is lifted from
    * ONNX code - looks like there could be a more C++ way to do this */
   FILE* f = fopen(filename.c_str(), "rb");
   if (f == NULL)
       return false;
   fseek(f, 0, SEEK_END);
   int size = ftell(f);
   fseek(f, 0, SEEK_SET);

   std::vector<char> data(size); // Use std::vector for dynamic memory allocation
   int nread = fread(data.data(), 1, size, f); // Use data.data() to access the underlying array
   fclose(f);

   if (nread != size)
       ERROR("Problem reading input data");

   ::google::protobuf::io::ArrayInputStream input_stream(data.data(), size); // Use data.data() here as well
   ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
   return result.ParseFromCodedStream(&coded_stream);
}
std::unique_ptr<OnnxConst> get_inputConst_from_file(std::string& partial_path, int input_number)
{
	onnx::TensorProto tensor;
	std::string input_fn = partial_path + std::to_string(input_number) + ".pb";

	if (load_input_data(input_fn, tensor) == false)
		return nullptr;

	std::unique_ptr<OnnxConst> tensorPointer = std::make_unique<OnnxConst>(tensor);
	return tensorPointer;
}

onnx::ValueInfoProto makeValueInfoFromTensorProto(const onnx::TensorProto& tensorProto)
{
	onnx::ValueInfoProto valueInfo;
	valueInfo.set_name(tensorProto.name());
	valueInfo.mutable_type()->mutable_tensor_type()->set_elem_type(tensorProto.data_type());
	for (const auto& dim : tensorProto.dims()) {
		valueInfo.mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
	}
	return valueInfo;
}

std::unique_ptr<OnnxVar> get_inputVar_from_file(std::string& partial_path, int input_number)
{
	onnx::TensorProto tensor;
	std::string input_fn = partial_path + std::to_string(input_number) + ".pb";

	if (load_input_data(input_fn, tensor) == false)
		return nullptr;
	onnx::ValueInfoProto valueInfo = makeValueInfoFromTensorProto(tensor);
	std::unique_ptr<OnnxVar> valueInfoPointer = std::make_unique<OnnxVar>(valueInfo);
	return valueInfoPointer;
}

int main(int argc, char* argv[])
{
	if (argc < 4) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << "./onnx_backend_tests_runner <directory> <accuracy> <test_data_set>" << std::endl;
		std::cerr << std::endl;
		std::cerr << " <directory> is the directory that contains the test - i.e. 'model.onnx' and test_data_set_0" << std::endl;
		std::cerr << " <accuracy> floating point value: the maximum allowed difference between result and refrence. Use decimal dot, not comma!" << std::endl;
		std::cerr << " <test_data_set> integer value: select the test dataset to run this test against. (Most tests have only 0)" << std::endl;
		exit(1);
	}
	std::cerr << "//Test" << std::endl;
	// Load the ONNX model
	onnx::ModelProto onnx_model;
	std::string dir(argv[1]);
	std::string model_fn = dir + "/model.onnx";
	onnx::LoadProtoFromPath(model_fn, onnx_model);
	// Print model to file specified by cmake
	std::string functionName = onnx2cpp::MakeCppFile(onnx_model, std::cout, true);
	// Print testsuite to get executed by CTest
	std::cout << "//Graph generated. All below is test suite code." << std::endl;

	std::vector<std::unique_ptr<OnnxConst>> inputs;
	std::vector<std::unique_ptr<OnnxVar>> outputs;
	std::vector<std::unique_ptr<OnnxConst>> references;
	std::string dataset_dir = dir + "/test_data_set_" + argv[3];

	int input_number = 0;
	while (true) {
		std::string partial = dataset_dir + "/input_";
		std::unique_ptr<OnnxConst> in = get_inputConst_from_file(partial, input_number);
		if (in == nullptr)
			break;
		in->Name(in->Name() + "_graph_in");
		inputs.push_back(std::move(in));
		input_number++;
	}

	input_number = 0;
	while (true) {
		std::string partial = dataset_dir + "/output_";
		std::unique_ptr<OnnxConst> ref = get_inputConst_from_file(partial, input_number);
		std::unique_ptr<OnnxVar> out = get_inputVar_from_file(partial, input_number);
		if (ref == nullptr || out == nullptr)
			break;
		ref->Name(ref->Name() + "_reference");
		out->Name(out->Name() + "_graph_out");
		references.push_back(std::move(ref));
		outputs.push_back(std::move(out));
		input_number++;
	}

	std::vector<std::string> inputNames;
	std::vector<std::string> outputNames;
	std::vector<std::string> referenceNames;




	std::cout << "int main(void) {" << std::endl;
	for (size_t i = 0; i < inputs.size(); ++i) {
		std::cout << inputs[i]->GetConstantString() << std::endl;
		inputNames.push_back(inputs[i]->Name());
	
	}

	for (size_t i = 0; i < outputs.size(); ++i) {
		std::cout << outputs[i]->GetVariableString() << std::endl;
		outputNames.push_back(outputs[i]->Name());
	}
	for (size_t i = 0; i < references.size(); ++i) {
		std::cout << references[i]->GetConstantString() << std::endl;
		referenceNames.push_back(references[i]->Name());
	}
	std::string hasInAndOut = "";
	if (!inputs.empty() && !outputs.empty()) {
		hasInAndOut = ", ";
	}
	std::cout << functionName << "(" << join(inputNames, ", ") << hasInAndOut << join(outputNames, ", ") << "); " << std::endl;
	for (size_t i = 0; i < outputs.size(); ++i) {
		if (i < references.size())
			std::cout << "if(" << outputNames[i] << "!=" << referenceNames[i] << ")" << std::endl;
		std::cout << "return 1; " << std::endl;
	}

	std::cout << "return 0;" << std::endl;
	std::cout << "}" << std::endl;

}