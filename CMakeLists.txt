# CMakeList.txt: CMake-Projektdatei der obersten Ebene. Führen Sie hier die globale Konfiguration aus,
# und schließen Sie Unterprojekte ein.
#
cmake_minimum_required (VERSION 3.8)

# Aktivieren Sie Hot Reload für MSVC-Compiler, sofern unterstützt.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("onnx2cpp")

# Schließen Sie Unterprojekte ein.
add_subdirectory ("onnx2cpp")

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(src)

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
	COMMAND protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/ --cpp_out=. ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/onnx.proto
	MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/onnx.proto
	)
