#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/cilk-kitsune/artifact-env.sh"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}

if [[ $(realpath -m -- "$OPENCILK_PATH") == "$(realpath -m -- "$KITSUNE_PATH")" ]]; then
  printf 'Set OPENCILK_PATH and KITSUNE_PATH to separate toolchain prefixes.\n' >&2
  exit 1
fi

if [[ -n ${OPENCILK_SOURCE:-} ]]; then
  OPENCILK_SOURCE=$(realpath -ms -- "$OPENCILK_SOURCE")
  for component in llvm cheetah cilktools; do
    if [[ ! -f "$OPENCILK_SOURCE/$component/CMakeLists.txt" ]]; then
      printf 'OpenCilk source is missing %s/CMakeLists.txt under %s. Set OPENCILK_SOURCE to the prepared source tree described in cilk-kitsune/OPENCILK.md.\n' "$component" "$OPENCILK_SOURCE" >&2
      exit 1
    fi
  done
else
  require_opencilk_frontend
fi

common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
        "-DCMAKE_C_COMPILER=$(command -v "${CC:-clang}")"
        "-DCMAKE_CXX_COMPILER=$(command -v "${CXX:-clang++}")")
if [[ -n ${LLVM_USE_LINKER:-} ]]; then
  common+=("-DLLVM_USE_LINKER=$LLVM_USE_LINKER")
fi

if [[ -n ${OPENCILK_SOURCE:-} ]]; then
  cmake -S "$OPENCILK_SOURCE/llvm" -B "$OPENCILK_PATH" "${common[@]}" \
    '-DLLVM_ENABLE_PROJECTS=clang;compiler-rt' \
    '-DLLVM_ENABLE_RUNTIMES=cheetah;cilktools' -DLLVM_TARGETS_TO_BUILD=X86
  cmake --build "$OPENCILK_PATH" --target clang clang-resource-headers cheetah
  cmake --build "$OPENCILK_PATH/runtimes/runtimes-bins" --target cilk-headers cilktools-headers
  require_opencilk_frontend
fi

cmake -S "$root/llvm" -B "$KITSUNE_PATH" "${common[@]}" \
  '-DLLVM_TARGETS_TO_BUILD=X86;NVPTX' '-DKITSUNE_ENABLED_TAPIR_TARGETS=cuda;opencilk' \
  -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DKITSUNE_KOKKOS_ENABLE=OFF \
  -DKITCUDA_ENABLE_NVTX=OFF "-DCUDAToolkit_ROOT=$CUDA_HOME"
cmake --build "$KITSUNE_PATH" --target clang clang-resource-headers kitsune-resource-headers opt llvm-link kitcc kit++ \
  cheetah-copy-to-build-dir kitrt-copy-to-build-dir
