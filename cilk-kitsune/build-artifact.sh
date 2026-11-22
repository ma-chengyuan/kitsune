#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
build=${BUILD_DIR:-"$root/build"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}

common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
        "-DCMAKE_C_COMPILER=$(command -v "${CC:-clang}")"
        "-DCMAKE_CXX_COMPILER=$(command -v "${CXX:-clang++}")")
if [[ -n ${LLVM_USE_LINKER:-} ]]; then
  common+=("-DLLVM_USE_LINKER=$LLVM_USE_LINKER")
fi

cmake -S "$root/opencilk-project/llvm" -B "$build/opencilk" "${common[@]}" \
  '-DLLVM_ENABLE_PROJECTS=clang;compiler-rt' \
  '-DLLVM_ENABLE_RUNTIMES=cheetah;cilktools' -DLLVM_TARGETS_TO_BUILD=X86
cmake --build "$build/opencilk" --target clang clang-resource-headers cheetah
cmake --build "$build/opencilk/runtimes/runtimes-bins" --target cilk-headers cilktools-headers

cmake -S "$root/llvm" -B "$build/kitsune" "${common[@]}" \
  '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX' '-DKITSUNE_ENABLED_TAPIR_TARGETS=cuda;opencilk' \
  -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DKITSUNE_KOKKOS_ENABLE=OFF \
  -DKITCUDA_ENABLE_NVTX=OFF "-DCUDAToolkit_ROOT=$CUDA_HOME"
cmake --build "$build/kitsune" --target clang clang-resource-headers kitsune-resource-headers opt llvm-link kitcc kit++ \
  cheetah-copy-to-build-dir kitrt-copy-to-build-dir
