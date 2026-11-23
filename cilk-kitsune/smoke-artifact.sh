#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
source "$here/artifact-env.sh"
if [[ ! -f "$KITSUNE_PATH/bin/clang++" || ! -x "$KITSUNE_PATH/bin/clang++" ]]; then
  printf 'Kitsune compiler is missing or not executable at %s. Set KITSUNE_PATH to its build prefix or run cilk-kitsune/build-artifact.sh.\n' "$KITSUNE_PATH/bin/clang++" >&2
  exit 1
fi

export CILK_NWORKERS=${CILK_NWORKERS:-2}
arch=${CUDA_ARCH:-sm_80}
clangxx="$KITSUNE_PATH/bin/clang++"
mixed=("$clangxx" -fzera "--cuda-path=$CUDA_HOME" "--cuabi-arch=$arch")
mkdir -p "$build/smoke"
out=$(mktemp -d "$build/smoke/run.XXXXXX")
echo "Smoke test outputs in $out"
cd "$out"

# Exercise both a source-plus-object link and an object-only link.
echo 'int answer() { return 42; }' > answer.cpp
echo 'int answer(); int main() { return answer() == 42 ? 0 : 1; }' > main.cpp
"${mixed[@]}" -O2 -std=c++17 -c answer.cpp -o answer.o
"${mixed[@]}" -O2 -std=c++17 main.cpp answer.o -o linked
./linked
"${mixed[@]}" -O2 -std=c++17 -c main.cpp -o main.o
"${mixed[@]}" main.o answer.o -o linked-objects
./linked-objects

"$clangxx" -std=c++17 -O2 -fopencilk \
  "$here/test_reducer_smoke.cpp" -o cpu-reducer
./cpu-reducer

for level in 0 2; do
  "$clangxx" -std=c++17 "-O$level" -fopencilk \
    "$here/test_cpu_cilk_smoke.cpp" -o "cpu-cilk-O$level"
  "./cpu-cilk-O$level"
done

# Check the executed GPU branch as well as its numerical result. A kitrt
# shutdown message is also printed by programs which used only a serial tail.
run_gpu_smoke() {
  local log=$1
  shift
  KITRT_VERBOSE=1 "$@" >"$log" 2>&1 || { cat "$log" >&2; return 1; }
  if ! grep -Eq "^kitcuda: (nested )?kernel '_cuabi_kern_[^']+' launch parameters:$" "$log"; then
    echo "No CUDA kernel launch observed in $log" >&2
    return 1
  fi
}

# Keep a lowered IR listing alongside each binary. Compile the executable from
# the source again so the hardware check uses a normal native driver invocation.
build_gpu_smoke() {
  local name=$1
  shift
  "${mixed[@]}" -O2 "$@" -S -emit-llvm -o "$name.ll"
  grep -Eq 'call .*@__kitcuda_launch_kernel\(' "$name.ll"
  "${mixed[@]}" -O2 "$@" -o "$name"
}

for mode in fused unfused naive; do
  flags=()
  case "$mode" in
    unfused) flags+=(--cuabi-dont-fuse-reducers) ;;
    naive) flags+=(--cuabi-naive-atomics) ;;
  esac
  build_gpu_smoke "$mode-reducer" -std=c++17 -DZERA_SMOKE_CUDA \
    "${flags[@]}" "$here/test_reducer_smoke.cpp"
  run_gpu_smoke "$mode.log" "./$mode-reducer"
done

build_gpu_smoke nested -std=c++17 "$here/test_parsum_2d.cpp"
run_gpu_smoke nested.log ./nested 17
grep -Fq 'Sum of first 17 numbers: 3648 vs 3648' nested.log

build_gpu_smoke transpose -g -std=c++17 "$here/test_transpose_smoke.cpp"
run_gpu_smoke transpose.log ./transpose

build_gpu_smoke scan -std=c++20 "$here/scan.cpp"
run_gpu_smoke scan.log ./scan 4097

resource=$("$KITSUNE_PATH/bin/clang++" -print-resource-dir)/lib
cilklib="$resource/x86_64-unknown-linux-gnu"
"$clangxx" "$here/test_prefetch_smoke.cpp" -o prefetch \
  -I"$CUDA_HOME/include" -L"$resource" -L"$cilklib" "-Wl,-rpath,$resource:$cilklib" \
  -lkitrt -lopencilk -lcuda
KITCUDA_BIG_BLOCK_THRESHOLD=4096 ./prefetch

if [[ -n ${ZERA_BENCHMARKS:-} || -n ${ICICLE_INCLUDE:-} ]]; then
  : "${ZERA_BENCHMARKS:?Set to the absolute path of zera-paper-benchmarks}"
  : "${ICICLE_INCLUDE:?Set to the absolute path of the patched Icicle include directory}"
  export LD_LIBRARY_PATH="$resource:$cilklib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$KITSUNE_PATH/bin/clang++" -std=c++20 -O2 -ftapir=cuda -mllvm "-cuabi-arch=$arch" \
    -mllvm "-kitcuda-runtime-bc-path=$resource/kitcuda.bc" -mllvm -cuabi-embed-ptx \
    -I"$ZERA_BENCHMARKS/include" -I"$ICICLE_INCLUDE" "$ZERA_BENCHMARKS/ntt/zera_ntt_batched.cpp" \
    -o tiled-ntt -L"$cilklib" "-Wl,-rpath,$cilklib" -lopencilk -lopencilk-personality-cpp
  for log_n in 12 21; do
    run_gpu_smoke "ntt-$log_n.log" ./tiled-ntt "$log_n" 2 1
    grep -Fq '[batched] OK (2 transforms)' "ntt-$log_n.log"
  done
fi
