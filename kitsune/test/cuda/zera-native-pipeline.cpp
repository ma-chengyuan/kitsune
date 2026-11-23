// REQUIRES: kitsune-opencilk, nvptx-registered-target, x86-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && env TMPDIR=%t CUDAABI_TARGET=sm_90 %kitxx --target=x86_64-unknown-linux-gnu \
// RUN:   -fzera -O2 -g -S -emit-llvm --cuabi-arch=sm_80 -mllvm -cuabi-keep-files %s -o host.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t/host.ll \
// RUN:   --implicit-check-not='detach within' --implicit-check-not='sync within'
// RUN: cat %t/*.ptx | FileCheck %s --check-prefix=PTX

extern "C" void cpu_gpu(int *out, int rows, int cols) {
  _Cilk_for (int row = 0; row < rows; ++row) {
    [[tapir::target("cuda")]]
    _Cilk_for (int col = 0; col < cols; ++col)
      out[row * cols + col] = row + col;
  }
}

extern "C" void copy(void *dst, const void *src, unsigned long n) {
  __builtin_memcpy(dst, src, n);
}

// HOST: @_cuabi_fatbin_ptr = private {{.*}}constant
// HOST-LABEL: define {{.*}}void @cpu_gpu(
// HOST: call fastcc void @[[HELPER:cpu_gpu\.outline[^ (]+]](
// HOST-LABEL: define {{.*}}void @copy(
// HOST: call void @__kitcuda_memcpy(
// HOST: define internal fastcc void @[[HELPER]](
// HOST: call ptr @__kitcuda_launch_kernel(
// HOST: !DISubprogram(name: "cpu_gpu"

// PTX: .target sm_80
// PTX: .entry _cuabi_kern_
