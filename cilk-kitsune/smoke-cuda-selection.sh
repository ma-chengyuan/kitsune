#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <clang++> <configured CUDA toolkit root> <competing CUDA toolkit root>" >&2
  exit 1
fi

clangxx=$1
selected=$(realpath -- "$2")
competing=$(realpath -- "$3")
export LC_ALL=C

cuda_soname() {
  readelf -dW "$1/lib64/libcudart.so" | sed -nE 's/.*\(SONAME\).* \[([^]]+)\].*/\1/p'
}

expected=$(cuda_soname "$selected")
other=$(cuda_soname "$competing")
if [[ -z $expected || -z $other || $expected == "$other" ]]; then
  echo 'Use two CUDA SDKs with different libcudart SONAMEs.' >&2
  exit 1
fi

out=$(mktemp -d)
trap 'rm -rf -- "$out"' EXIT
echo 'int main() {}' | "$clangxx" -x c++ -c - -o "$out/main.o"

linkers=(default)
for linker in bfd lld; do
  linker_path=$("$clangxx" "-print-prog-name=ld.$linker")
  if command -v "$linker_path" >/dev/null; then
    linkers+=("$linker")
  fi
done

for linker in "${linkers[@]}"; do
  flags=(-fzera -Wl,--as-needed "-L$competing/lib64")
  if [[ $linker != default ]]; then
    flags+=("-fuse-ld=$linker")
  fi
  for mode in explicit configured; do
    selection=()
    if [[ $mode == explicit ]]; then
      selection=("--cuda-path=$selected")
    fi

    executable="$out/$linker-$mode"
    "$clangxx" "${flags[@]}" "${selection[@]}" "$out/main.o" -o "$executable"
    dynamic=$(readelf -dW "$executable")
    actual=$(sed -nE 's/.*\(NEEDED\).* \[(libcudart\.so[^]]*)\].*/\1/p' <<<"$dynamic")
    if [[ $actual != "$expected" ]]; then
      printf '%s linker with %s toolkit selection expected %s.\n%s\n' \
        "$linker" "$mode" "$expected" "$dynamic" >&2
      exit 1
    fi
    printf '%s linker with %s toolkit selection uses %s\n' "$linker" "$mode" "$actual"
  done
done
