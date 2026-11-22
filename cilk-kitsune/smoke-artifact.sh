#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
build=$(cd "${BUILD_DIR:-$here/../build}" && pwd)
export OPENCILK_PATH="$build/opencilk" KITSUNE_PATH="$build/kitsune"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export CILK_NWORKERS=${CILK_NWORKERS:-2}
arch=${CUDA_ARCH:-sm_80}
mkdir -p "$build/smoke"
out=$(mktemp -d "$build/smoke/run.XXXXXX")
echo "Smoke test outputs in $out"
cd "$out"

"$OPENCILK_PATH/bin/clang++" -std=c++17 -O2 -fopencilk \
  "$here/test_reducer_smoke.cpp" -o cpu-reducer
./cpu-reducer

for mode in fused unfused; do
  flags=()
  if [[ $mode == unfused ]]; then flags+=(--cuabi-dont-fuse-reducers); fi
  "$here/ock++" -O2 -std=c++17 -fopencilk -DZERA_SMOKE_CUDA "--cuabi-arch=$arch" \
    "-save-temps=$out/$mode" "${flags[@]}" "$here/test_reducer_smoke.cpp" -o "$mode-reducer"
  grep -Eq 'call .*@__kitcuda_launch_kernel\(' "$mode/test_reducer_smoke.kitsune.ll"
  "./$mode-reducer"
done

"$here/ock++" -O2 -std=c++17 -fopencilk "--cuabi-arch=$arch" \
  "$here/test_parsum_2d.cpp" -o nested
./nested 17 | tee nested.log
grep -Fq 'Sum of first 17 numbers: 3648 vs 3648' nested.log

"$here/ock++" -g -O2 -std=c++17 "--cuabi-arch=$arch" \
  "$here/test_transpose_smoke.cpp" -o transpose
./transpose

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
    ./tiled-ntt "$log_n" 2 1 2>&1 | tee "ntt-$log_n.log"
    grep -Fq '[batched] OK (2 transforms)' "ntt-$log_n.log"
  done
fi
