# CMakeList.txt: CMake-Projekt für "onnx2cpp". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required(VERSION 3.20)
project(onnx2cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx)


# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
#add_executable (onnx2cpp "onnx2cpp.cpp" "onnx2cpp.h")


s#et_property(TARGET onnx2cpp PROPERTY CXX_STANDARD 20)




# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
