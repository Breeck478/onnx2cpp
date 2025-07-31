cmake_minimum_required(VERSION 3.28)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../targetIncludes)
set(ONNX_BACKEND_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test/data/)
set(ONNX_NODE_TEST_DATA_DIR ${ONNX_BACKEND_TEST_DATA_DIR}/node/)

find_package(onnxruntime CONFIG REQUIRED)
find_package(xtensor REQUIRED)

add_executable( testgen
	TestInitializer.cpp)
target_link_libraries(testgen PRIVATE onnx2cpp_lib ONNX::onnx ONNX::onnx_proto onnxruntime::onnxruntime)
target_include_directories(testgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)



function( ONNX_type_test node_name data_dir test_ctest_name accuracy test_data_set)

	set( test_c  ${node_name}_${test_data_set}_test.cpp )
	set( testbin ${node_name}_${test_data_set}_test )
	add_custom_command(
		OUTPUT
		${test_c}
		COMMAND
		testgen ${data_dir} ${accuracy} ${test_data_set} > ${test_c}
		DEPENDS
		#TODO also depends on test data -> don't depend, always run
		testgen
		)

	add_executable( ${testbin}
		${test_c}
		)
	target_compile_options( ${testbin}
		PRIVATE
			-Wall #-Werror
			#TODO: space for output tensor is generated, but not used.
			#-Wno-unused-variable
		)
	target_link_libraries(${testbin} PRIVATE xtensor onnx2cpp_lib)
	target_include_directories( ${testbin} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude)


	# register with CTest
	add_test(NAME ${test_ctest_name}
		COMMAND ${testbin}
		)
endfunction()

function( ONNX_backend_node_test node_name)
	ONNX_type_test(
		${node_name}
		${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
		ONNX_backend_${node_name}
		0.00002
		0
	)
endfunction()
ONNX_backend_node_test(abs)
ONNX_backend_node_test(add)
ONNX_backend_node_test(argmax_default_axis_example)
ONNX_backend_node_test(argmin_default_axis_example)
ONNX_backend_node_test(cast_DOUBLE_to_FLOAT)
ONNX_backend_node_test(concat_1d_axis_0)
#ONNX_backend_node_test(constant)
ONNX_backend_node_test(div)
#ONNX_backend_node_test(equal)
ONNX_backend_node_test(gemm_all_attributes)
#ONNX_backend_node_test(greater)
ONNX_backend_node_test(identity)
ONNX_backend_node_test(matmul_2d)
ONNX_backend_node_test(matmul_3d)
ONNX_backend_node_test(matmul_4d)
ONNX_backend_node_test(mul)
#ONNX_backend_node_test(pow)
ONNX_backend_node_test(relu)
ONNX_backend_node_test(reshape_allowzero_reordered)
#ONNX_backend_node_test(shape)
ONNX_backend_node_test(sin)
ONNX_backend_node_test(sub)
ONNX_backend_node_test(tanh)
ONNX_backend_node_test(where_example)