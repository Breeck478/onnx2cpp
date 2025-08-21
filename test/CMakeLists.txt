cmake_minimum_required(VERSION 3.28)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNXRUNTIME_LIB ${CMAKE_CURRENT_BINARY_DIR}/../vcpkg_installed/x64-windows/lib/onnxruntime.lib)
set(ONNXRUNTIME_INC_DIR${CMAKE_CURRENT_BINARY_DIR}/../vcpkg_installed/x64-windows/include)
set(ONNX_NODE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test/data/node/)
set(ONNX_GRAPH_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../test/data/graph/)
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
	target_compile_definitions(testgenOrt PRIVATE -DORT_COMPARE=1)
endif()

if(DCO_AVAILABLE)
	# tangent mode dco tests
	add_executable(testgenDcoTangent dcoTangentTestInitializer.cpp)
	target_link_libraries(testgenDcoTangent PRIVATE ONNX::onnx ONNX::onnx_proto)
	target_include_directories(testgenDcoTangent PRIVATE ${DCO_INC_PATH}_tangent ) 
	# required for including Operator.cpp files and activating the mkro inside
	if(MSVC) 
		target_link_libraries(testgenDcoTangent PRIVATE onnx2cpp_lib)
		set_target_properties(testgenDcoTangent PROPERTIES
		LINK_FLAGS "/WHOLEARCHIVE:onnx2cpp_lib.lib")
	else()
		target_link_libraries(testgenDcoTangent 
			PRIVATE 
			"-Wl,--whole-archive" 
			onnx2cpp_lib 
			"-Wl,--no-whole-archive"
		)
	endif()

	# adjoint mode dco tests
	add_executable(testgenDcoAdjoint dcoAdjointTestInitializer.cpp)
	target_link_libraries(testgenDcoAdjoint PRIVATE ONNX::onnx ONNX::onnx_proto)
	target_include_directories(testgenDcoAdjoint PRIVATE ${DCO_INC_PATH}_adjoint ) 
	# required for including Operator.cpp files and activating the mkro inside
	if(MSVC) 
		target_link_libraries(testgenDcoAdjoint PRIVATE onnx2cpp_lib)
		set_target_properties(testgenDcoAdjoint PROPERTIES
		LINK_FLAGS "/WHOLEARCHIVE:onnx2cpp_lib.lib")
	else()
		target_link_libraries(testgenDcoAdjoint 
			PRIVATE 
			"-Wl,--whole-archive" 
			onnx2cpp_lib 
			"-Wl,--no-whole-archive"
		)
	endif()
	# Test makefile generation
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

function( node_dco_tangent node_name data_dir test_ctest_name abs_accuracy rel_accuracy test_data_set)

	set( test_c  ${node_name}_${test_data_set}_test_dco_tangent.cpp )
	set( testbin ${node_name}_${test_data_set}_test_dco_tangent )	
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${testbin})
	add_custom_target(always_generate_${node_name}_tangent ALL
		COMMAND testgenDcoTangent ${data_dir} ${abs_accuracy} ${rel_accuracy} ${test_data_set} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/${test_c}
	)
	add_custom_target(always_generate_${node_name}_tangent_makefile ALL
		COMMAND makegen ${testbin} ${ONNX_NODE_TEST_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/Makefile
	)
endfunction()

function( node_dco_adjoint node_name data_dir test_ctest_name abs_accuracy rel_accuracy test_data_set)

	set( test_c  ${node_name}_${test_data_set}_test_dco_adjoint.cpp )
	set( testbin ${node_name}_${test_data_set}_test_dco_adjoint )	
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${testbin})
	add_custom_target(always_generate_${node_name}_adjoint ALL
		COMMAND testgenDcoAdjoint ${data_dir} ${abs_accuracy} ${rel_accuracy} ${test_data_set} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/${test_c}
	)
	add_custom_target(always_generate_${node_name}_adjoint_makefile ALL
		COMMAND makegen ${testbin} ${ONNX_NODE_TEST_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${testbin}/Makefile
	)
endfunction()

function( graph_test_call graph_name data_dir test_ctest_name accuracy test_data_set)
	set( test_c  ${graph_name}_${test_data_set}_test_graph.cpp )
	set( testbin ${graph_name}_${test_data_set}_test_graph )	
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
	target_compile_options(${testbin} PRIVATE /bigobj)
	# register with CTest
	add_test(NAME ${test_ctest_name}
		COMMAND ${testbin}
		)
endfunction()

function( ONNX_backend_node_test node_name)
	set(options)
	set(oneValueArgs ACCURACY DATASET)
	set(multiValueArgs)
	cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
	if (${arg_ACCURACY})
		set(accuracy ${arg_ACCURACY})
	else()
		set(accuracy 0.1)
	endif()
	if (${arg_DATASET})
		set(dataset ${arg_DATASET})
	else()
		set(dataset 0)
	endif()
	ONNX_type_test(
		${node_name}
		${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
		ONNX_backend_${node_name}
		${accuracy}
		${dataset}
	)
endfunction()


function( ONNX_backend_node_test_onnxruntime node_name)
	if((CMAKE_SYSTEM_NAME STREQUAL "Windows") AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
		set(options)
		set(oneValueArgs ACCURACY DATASET)
		set(multiValueArgs)
		cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
		if (${arg_ACCURACY})
			set(accuracy ${arg_ACCURACY})
		else()
			set(accuracy 0.00002)
		endif()
		if (${arg_DATASET})
			set(dataset ${arg_DATASET})
		else()
			set(dataset 0)
		endif()
		ONNX_type_test_onnxruntime(
			${node_name}
			${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}_ort
			${accuracy}
			${dataset}
		)
	endif()
endfunction()

function( node_dco_tangent_test node_name)
	if(DCO_AVAILABLE)
		set(options)
		set(oneValueArgs ABS_ACCURACY REL_ACCURACY DATASET)
		set(multiValueArgs)
		cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
		if (${arg_ABS_ACCURACY})
			set(abs_accuracy ${arg_ABS_ACCURACY})
		else()
			set(abs_accuracy 0.00002)
		endif()
		if (${arg_REL_ACCURACY})
			set(rel_accuracy ${arg_REL_ACCURACY})
		else()
			set(rel_accuracy 0.1)
		endif()
		if (${arg_DATASET})
			set(dataset ${arg_DATASET})
		else()
			set(dataset 0)
		endif()
		node_dco_tangent(
			${node_name}
			${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}_dco
			${abs_accuracy}
			${rel_accuracy}
			${dataset}
		)
	endif()
endfunction()

function( node_dco_adjoint_test node_name)
	if(DCO_AVAILABLE)
		set(options)
		set(oneValueArgs ABS_ACCURACY REL_ACCURACY DATASET)
		set(multiValueArgs)
		cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
		if (${arg_ABS_ACCURACY})
			set(abs_accuracy ${arg_ABS_ACCURACY})
		else()
			set(abs_accuracy 0.00002)
		endif()
		if (${arg_REL_ACCURACY})
			set(rel_accuracy ${arg_REL_ACCURACY})
		else()
			set(rel_accuracy 0.1)
		endif()
		if (${arg_DATASET})
			set(dataset ${arg_DATASET})
		else()
			set(dataset 0)
		endif()
		node_dco_adjoint(
			${node_name}
			${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
			ONNX_backend_${node_name}_dco
			${abs_accuracy}
			${rel_accuracy}
			${dataset}
		)
	endif()
endfunction()

function(node_dco_tests node_name)
	node_dco_adjoint_test(${node_name} ${ARGN})
	node_dco_tangent_test(${node_name} ${ARGN})
endfunction()


function( graph_test graph_name)
	set(options)
	set(oneValueArgs ACCURACY DATASET)
	set(multiValueArgs)
	cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")
	if (${arg_ACCURACY})
		set(accuracy ${arg_ACCURACY})
	else()
		set(accuracy 0.1)
	endif()
	if (${arg_DATASET})
		set(dataset ${arg_DATASET})
	else()
		set(dataset 0)
	endif()
	graph_test_call(
		${graph_name}
		${ONNX_GRAPH_TEST_DATA_DIR}/test_${graph_name}
		ONNX_graph_${graph_name}
		${accuracy}
		${dataset}
	)
endfunction()




ONNX_backend_node_test(abs)
ONNX_backend_node_test(add)
ONNX_backend_node_test(argmax_default_axis_example)
ONNX_backend_node_test(argmax_default_axis_example_select_last_index)
ONNX_backend_node_test(argmax_default_axis_random)
ONNX_backend_node_test(argmax_default_axis_random_select_last_index)
ONNX_backend_node_test(argmax_keepdims_example)
ONNX_backend_node_test(argmax_keepdims_example_select_last_index)
ONNX_backend_node_test(argmax_keepdims_random)
ONNX_backend_node_test(argmax_keepdims_random_select_last_index)
ONNX_backend_node_test(argmax_negative_axis_keepdims_example)
ONNX_backend_node_test(argmax_negative_axis_keepdims_example_select_last_index)
ONNX_backend_node_test(argmax_negative_axis_keepdims_random)
ONNX_backend_node_test(argmax_negative_axis_keepdims_random_select_last_index)
ONNX_backend_node_test(argmax_no_keepdims_example)
ONNX_backend_node_test(argmax_no_keepdims_example_select_last_index)
ONNX_backend_node_test(argmax_no_keepdims_random)
ONNX_backend_node_test(argmax_no_keepdims_random_select_last_index)
ONNX_backend_node_test(argmin_default_axis_example)
ONNX_backend_node_test(argmin_default_axis_example_select_last_index)
ONNX_backend_node_test(argmin_default_axis_random)
ONNX_backend_node_test(argmin_default_axis_random_select_last_index)
ONNX_backend_node_test(argmin_keepdims_example)
ONNX_backend_node_test(argmin_keepdims_example_select_last_index)
ONNX_backend_node_test(argmin_keepdims_random)
ONNX_backend_node_test(argmin_keepdims_random_select_last_index)
ONNX_backend_node_test(argmin_negative_axis_keepdims_example)
ONNX_backend_node_test(argmin_negative_axis_keepdims_example_select_last_index)
ONNX_backend_node_test(argmin_negative_axis_keepdims_random)
ONNX_backend_node_test(argmin_negative_axis_keepdims_random_select_last_index)
ONNX_backend_node_test(argmin_no_keepdims_example)
ONNX_backend_node_test(argmin_no_keepdims_example_select_last_index)
ONNX_backend_node_test(argmin_no_keepdims_random)
ONNX_backend_node_test(argmin_no_keepdims_random_select_last_index)
ONNX_backend_node_test(cast_DOUBLE_to_FLOAT)
ONNX_backend_node_test(cast_FLOAT_to_DOUBLE)
#ONNX_backend_node_test(cast_FLOAT_to_STRING)
#ONNX_backend_node_test(cast_STRING_to_FLOAT)
ONNX_backend_node_test(concat_1d_axis_0)
ONNX_backend_node_test(concat_1d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_0)
ONNX_backend_node_test(concat_2d_axis_1)
ONNX_backend_node_test(concat_2d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_0)
ONNX_backend_node_test(concat_3d_axis_1)
ONNX_backend_node_test(concat_3d_axis_2)
ONNX_backend_node_test(concat_3d_axis_negative_1)
ONNX_backend_node_test(concat_3d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_negative_3)
ONNX_backend_node_test(constant)
ONNX_backend_node_test(constantofshape_float_ones)
ONNX_backend_node_test(constantofshape_int_shape_zero)
ONNX_backend_node_test(constantofshape_int_zeros)
ONNX_backend_node_test(div)
ONNX_backend_node_test(equal)
ONNX_backend_node_test(erf)
ONNX_backend_node_test(expand_dim_changed)
ONNX_backend_node_test(expand_dim_unchanged)
ONNX_backend_node_test(gather_0)
ONNX_backend_node_test(gather_1)
ONNX_backend_node_test(gather_2d_indices)
ONNX_backend_node_test(gemm_all_attributes)
ONNX_backend_node_test(gemm_alpha)
ONNX_backend_node_test(gemm_beta)
ONNX_backend_node_test(gemm_default_matrix_bias)
ONNX_backend_node_test(gemm_default_scalar_bias)
ONNX_backend_node_test(gemm_default_single_elem_vector_bias)
ONNX_backend_node_test(gemm_default_vector_bias)
ONNX_backend_node_test(gemm_default_zero_bias)
ONNX_backend_node_test(gemm_transposeA)
ONNX_backend_node_test(gemm_transposeB)
ONNX_backend_node_test(greater)
ONNX_backend_node_test(identity)
ONNX_backend_node_test(loop11)
#ONNX_backend_node_test(loop13_seq) # sequences are not supported by onnx2cpp
#ONNX_backend_node_test(loop16_seq_none) # sequences are not supported by onnx2cpp
ONNX_backend_node_test(matmul_2d)
ONNX_backend_node_test(matmul_3d)
ONNX_backend_node_test(matmul_4d)
ONNX_backend_node_test(mul)
ONNX_backend_node_test(neg)
ONNX_backend_node_test(neg_example)
ONNX_backend_node_test(pow)
ONNX_backend_node_test(reduce_mean_default_axes_keepdims_example)
ONNX_backend_node_test(reduce_mean_default_axes_keepdims_random)
ONNX_backend_node_test(reduce_mean_do_not_keepdims_example)
ONNX_backend_node_test(reduce_mean_do_not_keepdims_random)
ONNX_backend_node_test(reduce_mean_keepdims_example)
ONNX_backend_node_test(reduce_mean_keepdims_random)
ONNX_backend_node_test(reduce_mean_negative_axes_keepdims_example)
ONNX_backend_node_test(reduce_mean_negative_axes_keepdims_random)
ONNX_backend_node_test(relu)
ONNX_backend_node_test(reshape_allowzero_reordered)
ONNX_backend_node_test(reshape_extended_dims)
ONNX_backend_node_test(reshape_negative_dim)
ONNX_backend_node_test(reshape_negative_extended_dims)
ONNX_backend_node_test(reshape_one_dim)
ONNX_backend_node_test(reshape_reduced_dims)
ONNX_backend_node_test(reshape_reordered_all_dims)
ONNX_backend_node_test(reshape_reordered_last_dims)
ONNX_backend_node_test(reshape_zero_and_negative_dim)
ONNX_backend_node_test(reshape_zero_dim)
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
ONNX_backend_node_test(softmax_axis_0)
#ONNX_backend_node_test(softmax_axis_0_expanded) #this and other expanded tests use Operators not yet implementet in onnx2cpp
ONNX_backend_node_test(softmax_axis_1)
ONNX_backend_node_test(softmax_axis_2)
ONNX_backend_node_test(softmax_default_axis)
ONNX_backend_node_test(softmax_example)
ONNX_backend_node_test(softmax_large_number)
ONNX_backend_node_test(softmax_negative_axis)
# es sind noch nicht alle Tests für den expanded Modus eingefügt. Der Rest solte auch noch hinzugefügt werden
ONNX_backend_node_test(sub)
ONNX_backend_node_test(tanh)
ONNX_backend_node_test(transpose_all_permutations_0)
ONNX_backend_node_test(transpose_all_permutations_1)
ONNX_backend_node_test(transpose_all_permutations_2)
ONNX_backend_node_test(transpose_all_permutations_3)
ONNX_backend_node_test(transpose_all_permutations_4)
ONNX_backend_node_test(transpose_all_permutations_5)
ONNX_backend_node_test(transpose_default)
ONNX_backend_node_test(unsqueeze_axis_0)
ONNX_backend_node_test(unsqueeze_axis_1)
ONNX_backend_node_test(unsqueeze_axis_2)
ONNX_backend_node_test(unsqueeze_axis_3)
ONNX_backend_node_test(unsqueeze_negative_axes)
ONNX_backend_node_test(unsqueeze_three_axes)
ONNX_backend_node_test(unsqueeze_two_axes)
ONNX_backend_node_test(unsqueeze_unsorted_axes)
ONNX_backend_node_test(where_example)

# DCO tests
node_dco_tests(abs)
node_dco_tests(add)
#node_dco_tests(argmax_default_axis_example) gives back positions in Tensor. Does not have a differentiate method
#node_dco_tests(argmin_default_axis_example) gives back positions in Tensor. Does not have a differentiate method
#node_dco_tests(cast_DOUBLE_to_FLOAT) casting does not have a differentiate method
node_dco_tests(concat_1d_axis_0)
node_dco_tests(concat_1d_axis_negative_1)
node_dco_tests(concat_2d_axis_0)
node_dco_tests(concat_2d_axis_1)
node_dco_tests(concat_2d_axis_negative_1)
node_dco_tests(concat_2d_axis_negative_2)
node_dco_tests(concat_3d_axis_0)
node_dco_tests(concat_3d_axis_1)
node_dco_tests(concat_3d_axis_2)
node_dco_tests(concat_3d_axis_negative_1)
node_dco_tests(concat_3d_axis_negative_2)
node_dco_tests(concat_3d_axis_negative_3)
#node_dco_tests(constant)
node_dco_tests(div)
#node_dco_tests(equal) bool functions do not have a differentiate method
node_dco_tests(erf) # finite differencce is not exact enough for erf
node_dco_tests(expand_dim_changed)
node_dco_tests(expand_dim_unchanged)
node_dco_tests(gemm_all_attributes)
node_dco_tests(gemm_alpha)
node_dco_tests(gemm_beta)
node_dco_tests(gemm_default_matrix_bias)
node_dco_tests(gemm_default_scalar_bias)
node_dco_tests(gemm_default_single_elem_vector_bias)
node_dco_tests(gemm_default_vector_bias)
node_dco_tests(gemm_default_zero_bias)
#node_dco_tests(greater) bool functions do not have a differentiate method
node_dco_tests(identity)
node_dco_tests(loop11)
node_dco_tests(matmul_2d)
node_dco_tests(matmul_3d)
node_dco_tests(matmul_4d)
node_dco_tests(mul)
node_dco_tests(pow REL_ACCURACY 0.5) # finite differencce is not exact enough for pow
node_dco_tests(relu)
#node_dco_tests(reshape_allowzero_reordered)
#node_dco_tests(shape) shape does not have a differentiate method because the return type is always vector of integers
node_dco_tests(sin)
node_dco_tests(slice)
node_dco_tests(slice_default_axes)
node_dco_tests(slice_default_steps)
node_dco_tests(slice_end_out_of_bounds)
node_dco_tests(slice_start_out_of_bounds)
node_dco_tests(slice_neg)
node_dco_tests(slice_neg_steps)
node_dco_tests(slice_negative_axes)
node_dco_tests(sub)
node_dco_tests(tanh)
node_dco_tests(unsqueeze_axis_0)
node_dco_tests(unsqueeze_axis_1)
node_dco_tests(unsqueeze_axis_2)
node_dco_tests(unsqueeze_axis_3)
node_dco_tests(unsqueeze_negative_axes)
node_dco_tests(unsqueeze_three_axes)
node_dco_tests(unsqueeze_two_axes)
node_dco_tests(unsqueeze_unsorted_axes)
#node_dco_tests(where_example) does not work because only one input of dco Type. Therefor cant combine Inputs because of different types 

graph_test(PINN)
graph_test(loopModel)
#graph_test(gptneox_Opset16)

