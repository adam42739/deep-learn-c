# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (deeplearnc)

set(CDEEPLEARN_SRC ${CDEEPLEARN_SRC}
	tests/main.c
	tests/tests.c
	src/general_lib/stdlib_mod.c
	src/general_lib/random.c
	src/general_ml/activation.c
	src/general_ml/loss.c
	src/layers/convolutional/convolutional_filter.c
	src/layers/convolutional/convolutional_layer.c
	src/layers/convolutional/image_structures.c
	src/layers/linear/linear_layer.c
	src/layers/linear/linear_model.c
	src/layers/layer.c
	src/linear_network.c
	src/vanilla_sgd.c
)
add_executable(deeplearnc ${CDEEPLEARN_SRC} )
