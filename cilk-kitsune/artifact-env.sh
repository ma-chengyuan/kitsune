# Common path setup for the build and smoke scripts. They set root before sourcing.
build=$(realpath -ms -- "${BUILD_DIR:-$root/build}")
OPENCILK_PATH=$(realpath -ms -- "${OPENCILK_PATH:-$build/opencilk}")
KITSUNE_PATH=$(realpath -ms -- "${KITSUNE_PATH:-$build/kitsune}")
CUDA_HOME=$(realpath -ms -- "${CUDA_HOME:-/usr/local/cuda}")
export OPENCILK_PATH KITSUNE_PATH CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

require_opencilk_frontend() {
  local clang="$OPENCILK_PATH/bin/clang++"
  if [[ ! -f "$clang" || ! -x "$clang" ]]; then
    printf 'OpenCilk frontend is missing or not executable at %s. Set OPENCILK_PATH to the compatible build or install prefix, or build from OPENCILK_SOURCE. See cilk-kitsune/ARTIFACT.md.\n' "$clang" >&2
    exit 1
  fi
  if ! "$clang" -### -x c++ -fopencilk-kitsune -ftapir=none -S -emit-llvm - >/dev/null 2>&1; then
    printf 'OpenCilk frontend at %s failed the -fopencilk-kitsune -ftapir=none driver probe. Check OPENCILK_PATH against the frontend revisions in cilk-kitsune/OPENCILK.md.\n' "$clang" >&2
    exit 1
  fi
}
