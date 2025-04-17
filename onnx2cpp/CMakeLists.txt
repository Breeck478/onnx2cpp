# CMakeList.txt: CMake-Projekt für "onnx2cpp". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required(VERSION 3.20)
project(onnx2cpp)
#find_package(ONNX CONFIG REQUIRED)
#find_package(protobuf CONFIG REQUIRED)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (onnx2cpp "onnx2cpp.cpp" "onnx2cpp.h")


set_property(TARGET onnx2cpp PROPERTY CXX_STANDARD 20)



#target_link_libraries(onnx2cpp PRIVATE ONNX::onnx_pb)
#target_link_libraries(onnx2cpp PRIVATE protobuf::protoc)
# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
