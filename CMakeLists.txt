# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
#
cmake_minimum_required (VERSION 3.24)

# Aktivieren Sie Hot Reload für MSVC-Compiler, sofern unterstützt.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message(WARNING "CMAKE_TOOLCHAIN_FILE nicht gesetzt – vcpkg wird nicht verwendet!")
endif()
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (onnx2cpp)
set(CMAKE_BUILD_TYPE Debug)
include_directories(src)
aux_source_directory(./src/Operators SOURCES)
message(WARNING ${SOURCES})
add_executable (onnx2cpp src/onnx2cpp.cpp src/OnnxVars.cpp src/Utils.cpp src/OnnxNodes.cpp src/OnnxConsts.cpp ${SOURCES}  "src/Operators/Gemm.cpp" "src/Operators/ConstantOfShape.cpp""src/Operators/Concat.cpp" "src/Operators/Pow.cpp" "src/Operators/Constant.cpp" "src/OnnxTensor.cpp" "src/Operators/Equal.cpp" "src/Operators/Where.cpp" "src/Operators/Greater.cpp" "src/OnnxGraph.cpp" "src/Operators/Loop.cpp" "src/Operators/Cast.cpp")

find_package(ONNX CONFIG REQUIRED)
target_link_libraries(onnx2cpp PRIVATE ONNX::onnx ONNX::onnx_proto)
find_package(xtensor REQUIRED)
target_link_libraries(onnx2cpp PRIVATE xtensor)
#set_property(TARGET onnx2cpp/onnx2cpp PROPERTY CXX_STANDARD 20)
#add_subdirectory(protobuf)
# Schließen Sie Unterprojekte ein'.
#add_subdirectory(onnx)
#
#include_directories(${Protobuf_INCLUDE_DIRS})
#include_directories(onnx2cpp)
#include_directories(Protobuf)

