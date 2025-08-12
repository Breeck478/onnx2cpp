cmake_minimum_required(VERSION 3.28)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNXRUNTIME_LIB ${CMAKE_CURRENT_BINARY_DIR}/../vcpkg_installed/x64-windows/lib/onnxruntime.lib)
set(ONNXRUNTIME_INC_DIR${CMAKE_CURRENT_BINARY_DIR}/../vcpkg_installed/x64-windows/include)
set(ONNX_NODE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test/data/node/)
set(ONNX_NODE_TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude/dcoInclude/)
set(ONNX_NODE_TEST_MAKEFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude/)
	
if (MSVC) # only neccesarry for Visual Studio
	add_compile_options(/wd5045) # disable warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre is specified
endif()
find_package(xtensor CONFIG REQUIRED)
# Search dco directory
set(DCO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude/dcoInclude/dco)

if(IS_DIRECTORY ${DCO_DIR})
	message(STATUS "DCO_DIR is set to ${DCO_DIR}")
	set(DCO_INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude/dcoInclude)
	set(DCO_AVAILABLE 1)
else()
	set(DCO_AVAILABLE 0)
	message(WARNING "DCO_DIR is not set or does not exist, skipping DCO tests.")
endif()

message(STATUS "xtensor_INCLUDE_DIRS: ${xtensor_INCLUDE_DIRS}")
add_executable(testgen TestInitializer.cpp)
target_link_libraries(testgen PRIVATE ONNX::onnx ONNX::onnx_proto)
target_include_directories(testgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ) 
# required for including Operator.cpp files and activating the mkro inside
if(MSVC)
	target_link_libraries(testgen PRIVATE onnx2cpp_lib)
	set_target_properties(testgen PROPERTIES
	LINK_FLAGS "/WHOLEARCHIVE:onnx2cpp_lib.lib")
else()
	target_link_libraries(testgen 
		PRIVATE 
		"-Wl,--whole-archive" 
		onnx2cpp_lib 
		"-Wl,--no-whole-archive"
	)
endif()

 
if((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 8)) # onnxruntime only avvailable for Windows in 64 bit
	add_executable(testgenOrt TestInitializer.cpp)
	target_include_directories(testgenOrt PRIVATE ${ONNXRUNTIME_INC_DIR})
	target_link_libraries(testgenOrt PRIVATE onnx2cpp_lib ONNX::onnx ONNX::onnx_proto ${ONNXRUNTIME_LIB})
	target_include_directories(testgenOrt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ) 
	set_target_properties(testgenOrt PROPERTIES
    LINK_FLAGS "/WHOLEARCHIVE:onnx2cpp_lib.lib")
	target_compile_definitions(testgenOrt PRIVATE ORT_COMPARE=1)
endif()

if(DCO_AVAILABLE)
	add_executable(testgenDco dcoTangentTestInitializer.cpp)
	target_link_libraries(testgenDco PRIVATE ONNX::onnx ONNX::onnx_proto)
	target_include_directories(testgenDco PRIVATE ${DCO_INC_PATH} ) 
	# required for including Operator.cpp files and activating the mkro inside
	if(MSVC) 
		target_link_libraries(testgenDco PRIVATE onnx2cpp_lib)
		set_target_properties(testgenDco PROPERTIES
		LINK_FLAGS "/WHOLEARCHIVE:onnx2cpp_lib.lib")
	else()
		target_link_libraries(testgenDco 
			PRIVATE 
			"-Wl,--whole-archive" 
			onnx2cpp_lib 
			"-Wl,--no-whole-archive"
		)
	endif()
	add_executable(makegen TestMakeFileGenerator.cpp)
	add_custom_command(
    TARGET makegen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${ONNX_NODE_TEST_MAKEFILE_DIR}Makefile
            ${CMAKE_CURRENT_BINARY_DIR}/Makefile
)
endif()

function( ONNX_type_test node_name data_dir test_ctest_name accuracy test_data_set)
	set( test_c  ${node_name}_${test_data_set}_test.cpp )
	set( testbin ${node_name}_${test_data_set}_test )
	add_custom_command(
		OUTPUT
		${test_c}
		COMMAND
		testgen ${data_dir} ${accuracy} ${test_data_set} > ${test_c}
		DEPENDS
		testgen
		)
	add_executable( ${testbin}
		${test_c}
		)
		if(MSVC)
			target_compile_options(${testbin} PRIVATE /w)
		else()
			target_compile_options(${testbin} PRIVATE -w)
		endif()
	target_link_libraries(${testbin} PRIVATE xtensor)
	target_include_directories( ${testbin} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude)


	# register with CTest
	add_test(NAME ${test_ctest_name}
		COMMAND ${testbin}
		)
endfunction()

function( ONNX_type_test_onnxruntime node_name data_dir test_ctest_name accuracy test_data_set)
	set( test_c  ${node_name}_${test_data_set}_test_ort.cpp )
	set( testbin ${node_name}_${test_data_set}_test_ort )	
	add_custom_command(
		OUTPUT
		${test_c}
		COMMAND
		testgenOrt ${data_dir} ${accuracy} ${test_data_set} > ${test_c} 
		DEPENDS
		testgenOrt
		)
	add_executable( ${testbin}
		${test_c}
		)
		if(MSVC)
			target_compile_options(${testbin} PRIVATE /w)
		else()
			target_compile_options(${testbin} PRIVATE -w)
		endif()
	target_link_libraries(${testbin} PRIVATE xtensor)
	target_include_directories( ${testbin} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../TestInclude)

	# register with CTest
	add_test(NAME ${test_ctest_name}
		COMMAND ${testbin}
		)
endfunction()

function( ONNX_type_test_dco node_name data_dir test_ctest_name accuracy test_data_set)

	set( test_c  ${node_name}_${test_data_set}_test_dco.cpp )
	set( testbin ${node_name}_${test_data_set}_test_dco )	
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${testbin})
	add_custom_target(always_generate_${node_name} ALL
		COMMAND testgenDco ${data_dir} ${accuracy} ${test_data_set} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/${test_c}
	)
	add_custom_target(always_generate_${node_name}_makefile ALL
		COMMAND makegen ${testbin} ${ONNX_NODE_TEST_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/Makefile
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


function( ONNX_backend_node_test_onnxruntime node_name)
	if((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
		ONNX_type_test_onnxruntime(
			${node_name}
			${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}_ort
			0.00002
			0
		)
	endif()
endfunction()

function( ONNX_backend_node_test_dco node_name)
	if(DCO_AVAILABLE)
		set(options)
		set(oneValueArgs ACCURACY)
		set(multiValueArgs)
		cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
		if (${arg_ACCURACY})
			set(accuracy ${arg_ACCURACY})
		else()
			set(accuracy 0.1)
		endif()
		ONNX_type_test_dco(
			${node_name}
			${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}_dco
			${accuracy}
			0
		)
	endif()
endfunction()

ONNX_backend_node_test(abs)
ONNX_backend_node_test(add)
ONNX_backend_node_test(argmax_default_axis_example)
ONNX_backend_node_test(argmin_default_axis_example)
ONNX_backend_node_test(cast_DOUBLE_to_FLOAT)
#ONNX_backend_node_test(concat_1d_axis_0)
ONNX_backend_node_test(constant)
ONNX_backend_node_test(div)
ONNX_backend_node_test(equal)
ONNX_backend_node_test(gemm_all_attributes)
ONNX_backend_node_test(greater)
ONNX_backend_node_test(identity)
#ONNX_backend_node_test(loop11) // Values of input Tensor are not correct. Maybe ther is some difference between differnt protobuf versions it got stored with.
#ONNX_backend_node_test(loop13_seq) // sequences are not supported by onnx2cpp
#ONNX_backend_node_test(loop16_seq_none) // sequences are not supported by onnx2cpp
#ONNX_backend_node_test(matmul_2d)
#ONNX_backend_node_test(matmul_3d)
#ONNX_backend_node_test(matmul_4d)
ONNX_backend_node_test(mul)
ONNX_backend_node_test(pow)
ONNX_backend_node_test(relu)
#ONNX_backend_node_test(reshape_allowzero_reordered)
ONNX_backend_node_test(shape)
ONNX_backend_node_test(sin)
ONNX_backend_node_test(slice)
ONNX_backend_node_test(slice_default_axes)
ONNX_backend_node_test(slice_default_steps)
ONNX_backend_node_test(slice_end_out_of_bounds)
ONNX_backend_node_test(slice_start_out_of_bounds)
ONNX_backend_node_test(slice_neg)
ONNX_backend_node_test(slice_neg_steps)
ONNX_backend_node_test(slice_negative_axes)
ONNX_backend_node_test(sub)
ONNX_backend_node_test(tanh)
ONNX_backend_node_test(unsqueeze_axis_0)
ONNX_backend_node_test(unsqueeze_axis_1)
ONNX_backend_node_test(unsqueeze_axis_2)
#ONNX_backend_node_test(unsqueeze_axis_3) user older version of this operator, which is not supported by onnx2cpp
ONNX_backend_node_test(unsqueeze_negative_axes)
ONNX_backend_node_test(unsqueeze_three_axes)
ONNX_backend_node_test(unsqueeze_two_axes)
ONNX_backend_node_test(unsqueeze_unsorted_axes)
ONNX_backend_node_test(where_example)
ONNX_backend_node_test_onnxruntime(abs)
ONNX_backend_node_test_onnxruntime(add)
ONNX_backend_node_test_onnxruntime(argmax_default_axis_example)
ONNX_backend_node_test_onnxruntime(argmin_default_axis_example)
ONNX_backend_node_test_onnxruntime(cast_DOUBLE_to_FLOAT)
#ONNX_backend_node_test_onnxruntime(concat_1d_axis_0)
#ONNX_backend_node_test_onnxruntime(constant)
ONNX_backend_node_test_onnxruntime(div)
ONNX_backend_node_test_onnxruntime(equal)
ONNX_backend_node_test_onnxruntime(gemm_all_attributes)
ONNX_backend_node_test_onnxruntime(greater)
ONNX_backend_node_test_onnxruntime(identity)
#ONNX_backend_node_test_onnxruntime(matmul_2d)
#ONNX_backend_node_test_onnxruntime(matmul_3d)
#ONNX_backend_node_test_onnxruntime(matmul_4d)
ONNX_backend_node_test_onnxruntime(mul)
ONNX_backend_node_test_onnxruntime(pow)
ONNX_backend_node_test_onnxruntime(relu)
#ONNX_backend_node_test_onnxruntime(reshape_allowzero_reordered)
ONNX_backend_node_test_onnxruntime(shape)
ONNX_backend_node_test_onnxruntime(sin)
ONNX_backend_node_test_onnxruntime(slice_negative_axes)
ONNX_backend_node_test_onnxruntime(sub)
ONNX_backend_node_test_onnxruntime(tanh)
ONNX_backend_node_test_onnxruntime(where_example)

# DCO tests
ONNX_backend_node_test_dco(abs)
ONNX_backend_node_test_dco(add)
#ONNX_backend_node_test_dco(argmax_default_axis_example) gives back positions in Tensor. Does not have a differentiate method
#ONNX_backend_node_test_dco(argmin_default_axis_example) gives back positions in Tensor. Does not have a differentiate method
#ONNX_backend_node_test_dco(cast_DOUBLE_to_FLOAT) casting does not have a differentiate method
#ONNX_backend_node_test_dco(concat_1d_axis_0)
#ONNX_backend_node_test_dco(constant)
ONNX_backend_node_test_dco(div)
#ONNX_backend_node_test_dco(equal) bool functions do not have a differentiate method
ONNX_backend_node_test_dco(gemm_all_attributes)
ONNX_backend_node_test_dco(gemm_alpha)
ONNX_backend_node_test_dco(gemm_beta)
ONNX_backend_node_test_dco(gemm_default_matrix_bias)
ONNX_backend_node_test_dco(gemm_default_scalar_bias)
ONNX_backend_node_test_dco(gemm_default_single_elem_vector_bias)
ONNX_backend_node_test_dco(gemm_default_vector_bias)
ONNX_backend_node_test_dco(gemm_default_zero_bias)
#ONNX_backend_node_test_dco(greater) bool functions do not have a differentiate method
ONNX_backend_node_test_dco(identity)
#ONNX_backend_node_test_dco(matmul_2d)
#ONNX_backend_node_test_dco(matmul_3d)
#ONNX_backend_node_test_dco(matmul_4d)
ONNX_backend_node_test_dco(mul)
ONNX_backend_node_test_dco(pow ACCURACY 0.5)
ONNX_backend_node_test_dco(relu)
#ONNX_backend_node_test_dco(reshape_allowzero_reordered)
#ONNX_backend_node_test_dco(shape) shape does not have a differentiate method because the return type is always vector of integers
ONNX_backend_node_test_dco(sin)
ONNX_backend_node_test_dco(sub)
ONNX_backend_node_test_dco(tanh)
#ONNX_backend_node_test_dco(where_example) bool functions do not have a differentiate method


