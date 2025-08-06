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
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (onnx2cpp)
set(CMAKE_BUILD_TYPE Debug)
include_directories(src)
find_package(ONNX CONFIG REQUIRED)
find_package(xtensor REQUIRED)
aux_source_directory(./src/Operators OPERATORSOURCE)
add_executable (onnx2cpp src/Main.cpp
src/onnx2cpp.cpp 
src/OnnxVar.cpp
src/Utils.cpp 
src/OnnxNode.cpp
src/OnnxConst.cpp
src/OnnxTensor.cpp
src/OnnxGraph.cpp
${OPERATORSOURCE})

target_link_libraries(onnx2cpp PRIVATE ONNX::onnx ONNX::onnx_proto)
target_link_libraries(onnx2cpp PRIVATE xtensor)

add_library(onnx2cpp_lib STATIC
src/onnx2cpp.cpp
src/OnnxGraph.cpp
src/OnnxVar.cpp
src/Utils.cpp 
src/OnnxNode.cpp
src/OnnxConst.cpp
src/OnnxTensor.cpp
${OPERATORSOURCE})

target_link_libraries(onnx2cpp_lib PRIVATE ONNX::onnx ONNX::onnx_proto)
target_link_libraries(onnx2cpp_lib PRIVATE xtensor)


#if (--TEST)
	set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
	enable_testing()
	add_subdirectory(test)
#endif()


