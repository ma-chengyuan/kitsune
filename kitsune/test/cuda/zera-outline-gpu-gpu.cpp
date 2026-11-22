// REQUIRES: nvptx-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: %kitxx -ftapir=none -O1 -S -emit-llvm %s -o %t/input.ll
// RUN: cd %t && env TMPDIR=%t opt -passes='tapir-lowering<O1>' -verify-each \
// RUN:   -tapir-target=cuda -use-kitcuda-runtime-bc=false \
// RUN:   -cuabi-arch=sm_80 -cuabi-keep-files -S input.ll -o host.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t/host.ll \
// RUN:   --implicit-check-not='call{{.*}}@__kitcuda_launch_kernel'
// RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t/post-loop.ll
// RUN: cat %t/*.ptx | FileCheck %s --check-prefix=PTX --implicit-check-not='.entry'

extern "C" void gpu_gpu(int *out, int rows, int cols) {
  [[tapir::target("cuda")]]
  _kitsune_forall (int row = 0; row < rows; ++row) {
    [[tapir::target("cuda")]]
    _kitsune_forall (int col = 0; col < cols; ++col)
      out[row * cols + col] = row + col;
  }
}

// The host launches one kernel with one block per outer iteration.
// HOST-LABEL: define {{.*}} @gpu_gpu(
// HOST-SAME: i32 {{[^,]+}} %[[ROWS:[^,]+]],
// HOST: %[[BLOCKS:[^ ]+]] = zext {{(nneg )?}}i32 %[[ROWS]] to i64
// HOST: %[[STREAM:[^ ]+]] = call ptr @__kitcuda_launch_kernel_nested({{.*}}, i64 %[[BLOCKS]],
// HOST: call void @__kitcuda_sync_thread_stream(ptr %[[STREAM]])

// Inside the kernel, the outer loop uses the block ID. The inner loop starts
// at the thread ID, strides by the block width, and synchronizes the block.
// DEVICE-LABEL: define {{.*}} @_cuabi_kern_gpu_gpu_{{[0-9]+}}(
// DEVICE-SAME: !cuda.kernel.nested
// DEVICE: %[[BLOCK_ID:[^ ]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// DEVICE: %[[ROW_START:[^ ]+]] = zext i32 %[[BLOCK_ID]] to i64
// DEVICE: phi i64 [ %[[ROW_START]],
// DEVICE: %[[THREAD_ID:[^ ]+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// DEVICE: zext i32 %[[THREAD_ID]] to i64
// DEVICE: %[[WIDTH:[^ ]+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// DEVICE: %[[STRIDE:[^ ]+]] = zext i32 %[[WIDTH]] to i64
// DEVICE: add {{.*}}i64 {{.*}}, %[[STRIDE]]
// DEVICE: call void @llvm.nvvm.barrier0()
// DEVICE: store i32
// PTX: .entry _cuabi_kern_
