#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
source "$here/artifact-env.sh"
require_opencilk_frontend
if [[ ! -f "$KITSUNE_PATH/bin/clang++" || ! -x "$KITSUNE_PATH/bin/clang++" ]]; then
  printf 'Kitsune compiler is missing or not executable at %s. Set KITSUNE_PATH to its build prefix or run cilk-kitsune/build-artifact.sh.\n' "$KITSUNE_PATH/bin/clang++" >&2
  exit 1
fi

export CILK_NWORKERS=${CILK_NWORKERS:-2}
arch=${CUDA_ARCH:-sm_80}
mkdir -p "$build/smoke"
out=$(mktemp -d "$build/smoke/run.XXXXXX")
echo "Smoke test outputs in $out"
cd "$out"

# The CMake launcher must accept a source and an already compiled object in
# the same invocation. The main translation unit requires the object's symbol.
echo 'int answer() { return 42; }' > answer.cpp
echo 'int answer(); int main() { return answer() == 42 ? 0 : 1; }' > main.cpp
"$here/ock++" -O2 -std=c++17 -c answer.cpp -o answer.o
"$here/ock++" -O2 -std=c++17 main.cpp answer.o -o linked
./linked

"$OPENCILK_PATH/bin/clang++" -std=c++17 -O2 -fopencilk \
  "$here/test_reducer_smoke.cpp" -o cpu-reducer
./cpu-reducer

for level in 0 2; do
  "$OPENCILK_PATH/bin/clang++" -std=c++17 "-O$level" -fopencilk \
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

for mode in fused unfused naive; do
  flags=()
  case "$mode" in
    unfused) flags+=(--cuabi-dont-fuse-reducers) ;;
    naive) flags+=(--cuabi-naive-atomics) ;;
  esac
  "$here/ock++" -O2 -std=c++17 -fopencilk -DZERA_SMOKE_CUDA "--cuabi-arch=$arch" \
    "-save-temps=$out/$mode" "${flags[@]}" "$here/test_reducer_smoke.cpp" -o "$mode-reducer"
  grep -Eq 'call .*@__kitcuda_launch_kernel\(' "$mode/test_reducer_smoke.kitsune.ll"
  run_gpu_smoke "$mode.log" "./$mode-reducer"
done

"$here/ock++" -O2 -std=c++17 -fopencilk "--cuabi-arch=$arch" \
  "-save-temps=$out/nested-ir" "$here/test_parsum_2d.cpp" -o nested
grep -Eq 'call .*@__kitcuda_launch_kernel\(' nested-ir/test_parsum_2d.kitsune.ll
run_gpu_smoke nested.log ./nested 17
grep -Fq 'Sum of first 17 numbers: 3648 vs 3648' nested.log

"$here/ock++" -g -O2 -std=c++17 "--cuabi-arch=$arch" \
  "-save-temps=$out/transpose-ir" "$here/test_transpose_smoke.cpp" -o transpose
grep -Eq 'call .*@__kitcuda_launch_kernel\(' transpose-ir/test_transpose_smoke.kitsune.ll
run_gpu_smoke transpose.log ./transpose

"$here/ock++" -O2 -std=c++20 -fopencilk "--cuabi-arch=$arch" \
  "-save-temps=$out/scan-ir" "$here/scan.cpp" -o scan
grep -Eq 'call .*@__kitcuda_launch_kernel\(' scan-ir/scan.kitsune.ll
run_gpu_smoke scan.log ./scan 4097

resource=$("$KITSUNE_PATH/bin/clang++" -print-resource-dir)/lib
cilklib="$resource/x86_64-unknown-linux-gnu"
"$OPENCILK_PATH/bin/clang++" "$here/test_prefetch_smoke.cpp" -o prefetch \
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
