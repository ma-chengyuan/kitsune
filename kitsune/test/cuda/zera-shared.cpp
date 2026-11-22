// REQUIRES: nvptx-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: %kitxx -ftapir=none -O1 -DSHARED_BYTES=64 -S -emit-llvm %s -o %t/constant.ll
// RUN: cd %t && env TMPDIR=%t opt -passes='tapir-lowering<O1>' -verify-each \
// RUN:   -tapir-target=cuda -use-kitcuda-runtime-bc=false \
// RUN:   -cuabi-arch=sm_80 -cuabi-keep-files -disable-output constant.ll
// RUN: cat %t/*.ptx | FileCheck %s --check-prefix=PTX \
// RUN:   --implicit-check-not=__kitcuda_block_shared
// RUN: %kitxx -ftapir=none -O1 -DSHARED_BYTES=n -S -emit-llvm %s -o %t/dynamic.ll
// RUN: cd %t && env TMPDIR=%t not opt -passes='tapir-lowering<O1>' -verify-each \
// RUN:   -tapir-target=cuda -use-kitcuda-runtime-bc=false -cuabi-arch=sm_80 \
// RUN:   -disable-output dynamic.ll 2>&1 | FileCheck %s --check-prefix=ERROR

extern "C" void *__kitcuda_block_shared(unsigned long long);

extern "C" void shared(int *out, const int *in, int n) {
  [[tapir::target("cuda")]]
  _kitsune_forall (int i = 0; i < n; ++i) {
    auto *tile = static_cast<volatile int *>(__kitcuda_block_shared(SHARED_BYTES));
    tile[i % 16] = in[i];
    out[i] = tile[i % 16];
  }
}

// A constant size becomes a 16-byte-aligned static shared array. The volatile
// accesses keep both the shared store and the load visible in the PTX.
// PTX: .shared .align 16 .b8 __kitcuda_smem{{[^[]+}}[64];
// PTX: st.volatile.shared.u32
// PTX: ld.volatile.shared.u32

// ERROR: error: kitsune: __kitcuda_block_shared size must be a compile-time constant
