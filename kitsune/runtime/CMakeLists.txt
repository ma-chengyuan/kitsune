# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
cmake_minimum_required(VERSION 3.20.0)

project(kitrt C CXX)

include(FetchContent)

set(KITRT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(KITRT_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

# There is support for two different levels of debugging.  The general
# debug mode (KITRT_ENABLE_DEBUG) will enable runtime assertion checks
# that will cause the runtime to both check and abort on any error
# conditions from the underlying runtimes (e.g., CUDA, HIP, etc.).  The
# second option (KITRT_ENABLE_VERBOSE) will report runtime details and
# give a general feel for what is going on (memory allocations, kernel
# launches, etc.).
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose execution mode for debugging." OFF)

option(KITCUDA_ENABLE_NVTX "Enable NVTX profiling within the runtime" OFF)
set(KITCUDA_BITCODE_ARCH "sm_80" CACHE STRING "CUDA architecture for bitcode generation")

# Common headers and source files.
set(KITRT_HDRS
  kitrt.h
  debug.h
  memory_map.h)

set(KITRT_SRCS
  kitrt.cpp
  debug.cpp
  memory.cpp
  memory_map.cpp)

set(KITRT kitrt)

add_library(${KITRT} SHARED ${KITRT_SRCS})
set_property(TARGET ${KITRT} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Link our Clang-generated ThinLTO objects with the matching Kitsune LLD.
  target_link_options(${KITRT} PRIVATE "--ld-path=${LLVM_BINARY_DIR}/bin/ld.lld")
endif()

if (KITRT_ENABLE_DEBUG)
  target_compile_definitions(${KITRT} -D_KITRT_DEBUG_)
endif()

if (KITRT_ENABLE_VERBOSE)
  target_compile_definitions(${KITRT} -D_KITRT_VERBOSE_)
endif()

# Add parallel-hashmap dependency
FetchContent_Declare(
  parallel_hashmap
  URL https://github.com/greg7mdp/parallel-hashmap/archive/refs/tags/v2.0.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(parallel_hashmap)
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.88.0/boost-1.88.0-cmake.tar.xz
  URL_HASH SHA256=f48b48390380cfb94a629872346e3a81370dc498896f16019ade727ab72eb1ec
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(BOOST_INCLUDE_LIBRARIES assert config move intrusive)
FetchContent_MakeAvailable(Boost)

target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${KITRT} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib)

target_include_directories(${KITRT}
  BEFORE PRIVATE
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR})


# FIXME: We probably don't need to explicitly link pthread and rt. These are
# only necessary for older versions of glibc, but we might get some antique
# software stacks on HPC machines.
target_link_libraries(${KITRT}
  PUBLIC
  pthread
  rt
  dl)

target_link_libraries(${KITRT} PRIVATE phmap Boost::intrusive)

# The KITSUNE_CUDA_* variables are defined in ../CMakeLists.txt (i.e. in
# kitsune/CMakeLists.txt). They are passed in via CMake because the Kitsune
# runtime is treated as an "ExternalProject".
if (KITSUNE_CUDA_ENABLED)
  list(APPEND KITRT_HDRS
    cuda/kitcuda.h
    cuda/kitcuda_dylib.h)

  if(KITSUNE_OPENCILK_ENABLED)
    # Reuse the legacy adapter's C++ allocation replacements. They call only
    # kitrt's C ABI, including when a C driver links the Zera application.
    add_library(zera_alloc OBJECT
      ${CMAKE_CURRENT_SOURCE_DIR}/../../cilk-kitsune/kitcuda_malloc.cpp)
    target_compile_features(zera_alloc PRIVATE cxx_std_17)
    target_compile_options(zera_alloc PRIVATE -fno-exceptions)
    set_target_properties(zera_alloc PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION OFF
      POSITION_INDEPENDENT_CODE ON)
    install(FILES $<TARGET_OBJECTS:zera_alloc>
      DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib
      RENAME zera_alloc.o)
  endif()

  target_sources(${KITRT} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_CUDA_ENABLED)
  target_include_directories(${KITRT} SYSTEM PUBLIC ${KITSUNE_CUDA_INCLUDE_DIR})
  target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIBRARY_DIR})

  target_link_libraries(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIB_NVPTX_STATIC}
    ${KITSUNE_CUDA_LIB_CUDART}
    ${KITSUNE_CUDA_LIB_CUDA})

  if (KITCUDA_ENABLE_NVTX)
    target_compile_definitions(${KITRT} PUBLIC KITCUDA_ENABLE_NVTX)

    find_library(LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
      NO_DEFAULT_PATH)

    target_link_libraries(${KITRT} PUBLIC ${LIB_NVTOOLS_EXT})
  endif()

  find_program(LLVM_LINK llvm-link PATHS ${LLVM_BINARY_DIR}/bin NO_DEFAULT_PATH)
  find_program(LLVM_CLANGXX clang++ PATHS ${LLVM_BINARY_DIR}/bin NO_DEFAULT_PATH)

  set(KITCUDA_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/kitcuda.bc)
  set(KITCUDA_BC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/scan_inlined.cu
  )
  set(KITCUDA_INDIVIDUAL_BC_FILES "")
  # Loop through each source file and generate bitcode
  foreach(CU_SOURCE_FILE ${KITCUDA_BC_SOURCES})
    get_filename_component(CU_FILENAME ${CU_SOURCE_FILE} NAME_WE) # Get base name without extension
    set(CU_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/${CU_FILENAME}-cuda-nvptx64-nvidia-cuda-${KITCUDA_BITCODE_ARCH}.bc)
    add_custom_command(
      OUTPUT ${CU_BC_FILE}
      COMMAND ${LLVM_CLANGXX} -x cuda -O2 ${CU_SOURCE_FILE} -DGEN_BITCODE
                              --cuda-gpu-arch=${KITCUDA_BITCODE_ARCH} -c -emit-llvm
      DEPENDS ${CU_SOURCE_FILE}
      COMMENT "Generating CUDA bitcode for ${CU_FILENAME}.cu"
      VERBATIM
    )
    list(APPEND KITCUDA_INDIVIDUAL_BC_FILES ${CU_BC_FILE})
  endforeach()
  set(KITCUDA_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/kitcuda.bc)
  add_custom_command(
    OUTPUT ${KITCUDA_BC_FILE}
    COMMAND ${LLVM_LINK} -o ${KITCUDA_BC_FILE} ${KITCUDA_INDIVIDUAL_BC_FILES}
    DEPENDS ${KITCUDA_INDIVIDUAL_BC_FILES} # Depends on all intermediate files
    COMMENT "Linking CUDA bitcode modules into kitcuda.bc"
    VERBATIM
  )
  add_custom_target(kitcuda_bc_target ALL DEPENDS ${KITCUDA_BC_FILE})
  add_dependencies(${KITRT} kitcuda_bc_target)
  install(FILES ${KITCUDA_BC_FILE} DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)
endif()

if(KITSUNE_INCLUDE_TESTS AND KITSUNE_CUDA_ENABLED AND NOT CMAKE_CROSSCOMPILING)
  enable_testing()
  add_subdirectory(test)
endif()

# The KITSUNE_HIP_* variables are defined in ../CMakeLists.txt (i.e. in
# kitsune/CMakeLists.txt). They are passed in via CMake because the Kitsune
# runtime is treated as an "ExternalProject".
#
# Note on AMD HIP CMake details...  When is HIP not HIP?  When it is HIP...
# It is easy to confuse requirements for HIP as the HIP "language" vs. the
# HIP runtime.   The language has tangled dependencies within a LLVM context
# as they build around LLVM and Clang too.  Using some of the 'hip::host'
# targets are actually intended for the language and not runtime libraries.
# Unfortunately, these details seem to be a moving target with ecah release
# of rocm/hip...
if (KITSUNE_HIP_ENABLED)
  list(APPEND KITRT_HDRS
    hip/kithip.h
    hip/kithip_dylib.h)

  target_sources(${KITRT} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_HIP_ENABLED)
  target_compile_definitions(${KITRT} PUBLIC __HIP_PLATFORM_AMD__)
  target_include_directories(${KITRT} PUBLIC ${KITSUNE_HIP_INCLUDE_DIR})
  target_link_directories(${KITRT} PUBLIC ${KITSUNE_HIP_LIBRARY_DIR})
  target_link_libraries(${KITRT} PUBLIC ${KITSUNE_HIP_LIB_AMDHIP64})
endif()

# Find LLVM's libraries relative to libkitrt in both prefix layouts.
set_property(TARGET ${KITRT} APPEND PROPERTY
  INSTALL_RPATH "$ORIGIN/../../../")
set_property(TARGET ${KITRT} APPEND PROPERTY
  BUILD_RPATH "$ORIGIN/../../../")

install(TARGETS ${KITRT}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)

# KITSUNE FIXME: Do we really need to install the runtime headers? Do we intend
# this to be used by third-parties?
install(FILES ${KITRT_HDRS}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/include/kitrt)

# KITSUNE FIXME: Do we really need to install the headers for the targets?
install(DIRECTORY cuda hip realm
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/include/kitrt
  FILES_MATCHING PATTERN "*.h")
