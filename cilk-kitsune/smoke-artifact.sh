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

# Exercise final links with a source and an object, then with only objects.
# Neither input uses Cilk, and eager binding checks the runtime's dependencies.
echo 'int answer() { return 42; }' > answer.cpp
echo 'int answer(); int main() { return answer() == 42 ? 0 : 1; }' > main.cpp
"${mixed[@]}" -O2 -std=c++17 -c answer.cpp -o answer.o
"${mixed[@]}" -O2 -std=c++17 -c main.cpp -o main.o

# An ordinary C++ shared library must use the executable's managed allocator,
# even when the final Zera link hides symbols from static archives.
"$clangxx" -O2 -std=c++17 -shared -fPIC -Wl,-soname,libshared-allocation.so \
  "$here/test_shared_allocation/library.cpp" -o libshared-allocation.so
"$clangxx" -O2 -std=c++17 -I"$CUDA_HOME/include" -c \
  "$here/test_shared_allocation/main.cpp" -o shared-allocation.o

linkers=(default)
for linker in bfd lld; do
  linker_path=$("$clangxx" "-print-prog-name=ld.$linker")
  if command -v "$linker_path" >/dev/null; then
    linkers+=("$linker")
  fi
done
for linker in "${linkers[@]}"; do
  link_flags=(-Wl,--as-needed)
  if [[ $linker != default ]]; then
    link_flags+=("-fuse-ld=$linker")
  fi
  for input in main.cpp main.o; do
    name="as-needed-$linker-${input##*.}"
    "${mixed[@]}" -O2 -std=c++17 "${link_flags[@]}" "$input" answer.o -o "$name"
    LD_BIND_NOW=1 "./$name"
  done
  "${mixed[@]}" "${link_flags[@]}" -Wl,--exclude-libs,ALL '-Wl,-rpath,$ORIGIN' \
    shared-allocation.o ./libshared-allocation.so -o "shared-allocation-$linker"
  LD_BIND_NOW=1 "./shared-allocation-$linker"
done

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
gpu_launch_pattern="^kitcuda: (nested )?kernel '_cuabi_kern_[^']+' launch parameters:$"
run_gpu_smoke() {
  local log=$1
  shift
  KITRT_VERBOSE=1 "$@" >"$log" 2>&1 || { cat "$log" >&2; return 1; }
  if ! grep -Eq "$gpu_launch_pattern" "$log"; then
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

# Both translation units use the same CUDA kernel name. The debug build
# derives the name from the source basename and line, so keep both aligned.
for name in same-name same-name-debug; do
  flags=(-O2 -std=c++17)
  if [[ $name == same-name-debug ]]; then
    flags+=(-g)
  fi
  for module in a b; do
    "${mixed[@]}" "${flags[@]}" -c "$here/test_same_kernel_name/$module/update.cpp" \
      -o "$name-$module.o"
  done
  "${mixed[@]}" "${flags[@]}" "$here/test_same_kernel_name/main.cpp" \
    "$name-a.o" "$name-b.o" -o "$name"
  run_gpu_smoke "$name.log" "./$name"
  mapfile -t launches < <(grep -E "$gpu_launch_pattern" "$name.log")
  if [[ ${#launches[@]} -ne 2 || ${launches[0]} != "${launches[1]}" ]]; then
    echo "Expected two CUDA launches with the same kernel name in $name.log" >&2
    exit 1
  fi
done

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
