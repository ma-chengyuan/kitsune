find_package(Threads REQUIRED)

function(add_cuda_launch_cache_test test)
  add_executable(${test} cuda_launch_cache_test.cpp ../cuda/launching.cpp)
  target_compile_features(${test} PRIVATE cxx_std_17)
  target_include_directories(${test} PRIVATE .. ../cuda)
  target_include_directories(${test} SYSTEM PRIVATE ${KITSUNE_CUDA_INCLUDE_DIR})
  target_link_libraries(${test} PRIVATE phmap Threads::Threads)
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES TIMEOUT 30)
endfunction()

add_cuda_launch_cache_test(cuda-launch-cache-test)

if(KITRT_TEST_THREAD_SANITIZER)
  include(CheckCXXCompilerFlag)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=thread)
  check_cxx_compiler_flag(-fsanitize=thread KITRT_HAS_THREAD_SANITIZER)
  cmake_pop_check_state()
  if(NOT KITRT_HAS_THREAD_SANITIZER)
    unset(KITRT_HAS_THREAD_SANITIZER CACHE)
    message(FATAL_ERROR
      "KITRT_TEST_THREAD_SANITIZER requires the kitrt compiler to support "
      "building and linking with -fsanitize=thread")
  endif()

  add_cuda_launch_cache_test(cuda-launch-cache-test-tsan)
  target_compile_options(cuda-launch-cache-test-tsan PRIVATE
    -fsanitize=thread -g -O1)
  target_link_options(cuda-launch-cache-test-tsan PRIVATE -fsanitize=thread)
  set_tests_properties(cuda-launch-cache-test-tsan PROPERTIES
    ENVIRONMENT "TSAN_OPTIONS=halt_on_error=1")
endif()
