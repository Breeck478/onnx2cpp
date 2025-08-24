#include "onnx2cpp.h"
#include "Utils.h"
#include "OnnxConst.h"
#include "OnnxVar.h"
#include <iostream>
#ifdef ORT_COMPARE
#include <onnxruntime_cxx_api.h>
#include <experimental_onnxruntime_cxx_api.h>
#endif
#include <onnx/common/file_utils.h>
#include <onnx/version_converter/convert.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <memory>
#include <vector>
using namespace toCpp;

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
	   throw std::runtime_error("Problem reading input data");

   ::google::protobuf::io::ArrayInputStream input_stream(data.data(), size); 
   ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
   return result.ParseFromCodedStream(&coded_stream);
}

onnx::TensorProto get_inputProto_from_file(std::string& partial_path, int input_number)
{
	onnx::TensorProto tensor;
	std::string input_fn = partial_path + std::to_string(input_number) + ".pb";

	if (load_input_data(input_fn, tensor) == false)
		return onnx::TensorProto();

	return tensor;
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

#ifdef ORT_COMPARE
template <typename T>
Ort::Value MakeOrtValueFromDataAndShape(Ort::AllocatorWithDefaultOptions& allocator, std::vector<int64_t> shape, std::vector<T> data) {
	Ort::Value tensor = Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());
	T* tensor_data = tensor.GetTensorMutableData<T>();
	if constexpr (!std::is_same_v<T, bool>) {
		std::memcpy(tensor_data, data.data(), data.size() * sizeof(T));
	}
	else {
		for (size_t i = 0; i < data.size(); ++i) {
			tensor_data[i] = data[i] != 0;
		}
	}
	return tensor;
}

Ort::Value TensorProtoToOrtValue(const onnx::TensorProto& tensorProto, Ort::AllocatorWithDefaultOptions& allocator) {
	// Shape extrahieren
	std::vector<int64_t> shape;
	for (int64_t i = 0; i < tensorProto.dims_size(); i++) {
		shape.push_back(tensorProto.dims(i));
	}

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// Je nach Datentyp unterschiedlich behandeln
	switch (tensorProto.data_type()) {
	case onnx::TensorProto::FLOAT: {
		std::vector<float> data = ExtractDataFromTensor<float>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_INT64: {
		std::vector<int64_t> data = ExtractDataFromTensor<int64_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_INT32: {
		std::vector<int32_t> data = ExtractDataFromTensor<int32_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_INT16: {
		std::vector<int16_t> data = ExtractDataFromTensor<int16_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_INT8: {
		std::vector<int8_t> data = ExtractDataFromTensor<int8_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_UINT64: {
		std::vector<uint64_t> data = ExtractDataFromTensor<uint64_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_UINT32: {
		std::vector<uint32_t> data = ExtractDataFromTensor<uint32_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_UINT16: {
		std::vector<uint16_t> data = ExtractDataFromTensor<uint16_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_UINT8: {
		std::vector<uint8_t> data = ExtractDataFromTensor<uint8_t>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_DOUBLE: {
		std::vector<double> data = ExtractDataFromTensor<double>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	case onnx::TensorProto_DataType_STRING: {
		throw std::runtime_error("Error(TensorProtoToOrtValue): String type not supported by runtime "); // is not supported by ONNX Runtime. Dont know why. Could add new TypeToTensorType in onnxruntime_cxx_inline.h for string. But might be on purpose by ONNX?
	}
	case onnx::TensorProto_DataType_BOOL: {
		std::vector<bool> data = ExtractDataFromTensor<bool>(tensorProto);
		return MakeOrtValueFromDataAndShape(allocator, shape, data);
	}
	default:
		throw std::runtime_error("Error(TensorProtoToOrtValue): Not supported TensorProto datatype: " + GetDataTypeString(tensorProto.data_type()));
	}
}

template<typename T>
void ExtractDataFromOrtValue(onnx::TensorProto& tensorProto, const Ort::Value& ortTensor) {
	const T* data = ortTensor.GetTensorData<T>();
	size_t num_elements = ortTensor.GetTensorTypeAndShapeInfo().GetElementCount();
	std::string raw(reinterpret_cast<const char*>(data), num_elements * sizeof(T));
	tensorProto.set_raw_data(raw);
}


void SetTensorProtoFromOrtValue(onnx::TensorProto& tensorProto, const Ort::Value& ortTensor, const std::string name) {
	tensorProto.set_name(name);
	tensorProto.set_data_type(ortTensor.GetTensorTypeAndShapeInfo().GetElementType());
	std::vector<int64_t> shape = ortTensor.GetTensorTypeAndShapeInfo().GetShape();
	for (size_t j = 0; j < ortTensor.GetTensorTypeAndShapeInfo().GetShape().size(); ++j) {
		tensorProto.add_dims(shape[j]);
	}
	switch (ortTensor.GetTensorTypeAndShapeInfo().GetElementType()) {
	case onnx::TensorProto_DataType_FLOAT:
		ExtractDataFromOrtValue<float>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_INT64):
		ExtractDataFromOrtValue<int64_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_INT32):
		ExtractDataFromOrtValue<int32_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_INT16):
		ExtractDataFromOrtValue<int16_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_INT8):
		ExtractDataFromOrtValue<int8_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_UINT64):
		ExtractDataFromOrtValue<uint64_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_UINT32):
		ExtractDataFromOrtValue<uint32_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_UINT16):
		ExtractDataFromOrtValue<uint16_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_UINT8):
		ExtractDataFromOrtValue<uint8_t>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_DOUBLE):
		ExtractDataFromOrtValue<double>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_STRING):
		ExtractDataFromOrtValue<std::string>(tensorProto, ortTensor);
		break;
	case (onnx::TensorProto_DataType_BOOL):
		ExtractDataFromOrtValue<bool>(tensorProto, ortTensor);
		break;
	default:
		throw std::runtime_error("ERROR(SetTensorProtoFromOrtValue): Tensor data type " + GetDataTypeString(tensorProto.data_type()) + " not supported for Constant " + tensorProto.name());
	}


}

void get_ort_results(const std::string& modelPath, std::vector<onnx::TensorProto> inputs, std::vector<std::unique_ptr<OnnxConst>>& outputs)
{

#ifdef _WIN32
	std::string str = modelPath;
	std::wstring wide_string = std::wstring(str.begin(), str.end());
	std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
	std::string model_file = modelPath;
#endif
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ORTReferencesRun");
	Ort::SessionOptions session_options;
	std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
	Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options);
	std::vector<Ort::Value> ortInputs;
	Ort::AllocatorWithDefaultOptions allocator;
	for (size_t i = 0; i < inputs.size(); ++i) {
		try {
			Ort::Value inputTensor = TensorProtoToOrtValue(inputs[i], allocator);

		if (inputTensor.IsTensor()) {
			auto outputInfo = inputTensor.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> outputShape = outputInfo.GetShape();
			size_t num_elems = 1;
			for (auto d : outputShape) num_elems *= d; // Gesamtanzahl der Werte berechnen

			// Beispiel f�r float-Ausgabe:
			float* outputData = inputTensor.GetTensorMutableData<float>();
			std::vector<float> outputVec(outputData, outputData + num_elems);
		}
		ortInputs.push_back(std::move(inputTensor));
		}
		catch (const std::exception& e) {
			std::cerr << "ERROR(get_ort_results): Error converting input tensor: " << e.what() << std::endl;
			exit(1);
		}
	}
	try {
		auto output_tensors = session.Run(session.GetInputNames(), ortInputs, session.GetOutputNames());
		for (size_t i = 0; i < output_tensors.size(); ++i) {
			Ort::Value& output_tensor = output_tensors[i];
			// Convert Ort::Value to onnx::TensorProto
			onnx::TensorProto tensorProto;
			SetTensorProtoFromOrtValue(tensorProto, output_tensor, session.GetOutputNames()[i]);
			std::unique_ptr<OnnxConst> out = std::make_unique<OnnxConst>(tensorProto);
			outputs.push_back(std::move(out));
		}
	}
	catch (const Ort::Exception& e) {
		std::cerr << "ERROR(get_ort_results): Error during ONNX Runtime session run: " << e.what() << std::endl;
		exit(1);
	}
}

#endif
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
	//std::cout << "#include <xtensor/xio.hpp>" << std::endl;
	std::cout << "#pragma once" << std::endl;
	std::cout << "#include <xtensor.hpp>" << std::endl;
	// Load the ONNX model
	float accuracy = std::stod(argv[2]);
	onnx::ModelProto onnx_model;
	std::string dir(argv[1]);
	std::string model_fn = dir + "/model.onnx";
	onnx::LoadProtoFromPath(model_fn, onnx_model); 
	// convert Model to version supported by onnx2cpp
	onnx_model = onnx::version_conversion::ConvertVersion(onnx_model, 18);
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
#ifdef ORT_COMPARE
	// Get results from the ONNX Runtime
	try {
		std::cout << "// Getting results from ONNX Runtime" << std::endl;
		std::vector<onnx::TensorProto> protoInputs;
		input_number = 0;
		while (true) {
			std::string partial = dataset_dir + "/input_";
			onnx::TensorProto proto = get_inputProto_from_file(partial, input_number);
			if (proto.dims().size() == 0)
				break;
			protoInputs.push_back(proto);
			input_number++;
		}
		get_ort_results(model_fn, protoInputs, references);
		for (size_t i = 0; i < references.size(); ++i) {
			// create new valueinfo for each reference (output of model run)
			onnx::ValueInfoProto valueInfo;
			// Set the name type and shape of the valueInfo
			valueInfo.set_name(references[i]->Name()+"_out");
			auto* typeProto = valueInfo.mutable_type();
			auto* tensorType = typeProto->mutable_tensor_type();
			tensorType->set_elem_type(references[i]->DataType());
			auto* shapeProto = tensorType->mutable_shape();
			std::vector<int64_t> shape;
			for (size_t j = 0; j < references[i]->Shape().size(); j++)
				shapeProto->add_dim()->set_dim_value(references[i]->Shape()[j]);
			// create new OnnxVar from valueInfo
			outputs.push_back(std::make_unique<OnnxVar>(valueInfo));
		}
	}
	catch (const std::exception& e) {
		std::cout << "Error during ONNX Runtime processing: " << e.what() << std::endl;
		exit(1);
	}
#else
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
#endif
	std::vector<std::string> inputNames;
	std::vector<std::string> outputNames;
	std::vector<std::string> referenceNames;

	std::cout << "int main(void) {" << std::endl;
	std::cout << "try {" << std::endl;
	std::cout << "// Graph inputs" << std::endl;
	for (size_t i = 0; i < inputs.size(); ++i) {
		std::cout << inputs[i]->GetConstantString() << std::endl;
		inputNames.push_back(inputs[i]->Name());
	}
	std::cout << "// Graph outputs" << std::endl;
	for (size_t i = 0; i < outputs.size(); ++i) {
		std::cout << outputs[i]->GetVariableString() << std::endl;
		outputNames.push_back(outputs[i]->Name());
	}
	std::cout << "// Reference Values" << std::endl;



	for (size_t i = 0; i < references.size(); ++i) {
		std::cout << references[i]->GetConstantString() << std::endl;
		referenceNames.push_back(references[i]->Name());
	}
	std::string hasInAndOut = "";
	if (!inputs.empty() && !outputs.empty()) {
		hasInAndOut = ", ";
	}
	
	std::cout << functionName << "(" << Join(inputNames, ", ") << hasInAndOut << Join(outputNames, ", ") << "); " << std::endl;
	for (size_t i = 0; i < outputs.size(); ++i) {
		if (i < references.size()) {
			std::cout << "if(" << outputNames[i] << ".shape() !=" << referenceNames[i] << ".shape()){" << std::endl;
			std::cout << "std::cout << \"Test failed because shape is not equal for output-shape " << outputNames[i] << ".\\n Expected: \\n\" << xt::adapt(" << referenceNames[i] << ".shape()) << \"\\n Actual: \\n\" << xt::adapt(" << outputNames[i] << ".shape()) << std::endl;" << std::endl;
			std::cout << "return 1; " << std::endl;
			std::cout << "}" << std::endl;
			std::cout << "for(std::size_t i = 0; i < "<< outputNames[i]<< ".size(); ++i){" << std::endl;
			std::cout << "if(std::abs(" << outputNames[i] << ".flat(i) - " << referenceNames[i] << ".flat(i)) > "<< accuracy << "){" << std::endl;
			std::cout << "std::cout << \"Test failed for output " << outputNames[i] << ".\\n Expected: \\n\" << " << referenceNames[i] << " << \"\\n Actual: \\n\" << " << outputNames[i] << " << std::endl;" << std::endl;
			std::cout << "return 1; " << std::endl;
			std::cout << "}" << std::endl;
			std::cout << "}" << std::endl;
		}
	}
	std::cout << "} catch (const std::exception& e) {" << std::endl;
	std::cout << "std::cerr << \"Exception caught: \" << e.what() << std::endl;" << std::endl;
	std::cout << "return 1;" << std::endl;
	std::cout << "}" << std::endl;
	std::cout << "std::cout << \"Test passed!\" << std::endl;" << std::endl;
	std::cout << "return 0;" << std::endl;
	std::cout << "}" << std::endl;

}
