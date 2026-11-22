// REQUIRES: kitsune-opencilk, nvptx-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: %kitxx -ftapir=none -O1 -g -S -emit-llvm %s -o %t/input.ll
// RUN: cd %t && env TMPDIR=%t opt -passes='tapir-lowering<O1>' -verify-each \
// RUN:   -tapir-target=opencilk -use-opencilk-runtime-bc=false -debug-abi-calls \
// RUN:   -use-kitcuda-runtime-bc=false -cuabi-arch=sm_80 -cuabi-keep-files \
// RUN:   -S input.ll -o host.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t/host.ll \
// RUN:   --implicit-check-not='call{{.*}}@__kitcuda_launch_kernel'
// RUN: FileCheck %s --check-prefix=DEVICE --input-file=%t/post-loop.ll \
// RUN:   --implicit-check-not='!cuda.kernel.nested'
// RUN: cat %t/*.ptx | FileCheck %s --check-prefix=PTX --implicit-check-not='.entry'

extern "C" void cpu_gpu(int *out, int rows, int cols) {
  [[tapir::target("opencilk")]]
  _kitsune_forall (int row = 0; row < rows; ++row) {
    [[tapir::target("cuda")]]
    _kitsune_forall (int col = 0; col < cols; ++col)
      out[row * cols + col] = row + col;
  }
}

// The outlined CPU loop launches the inner GPU loop and waits for it.
// HOST-LABEL: define {{.*}} @cpu_gpu(
// HOST: call fastcc void @[[HELPER:cpu_gpu\.outline[^ (]+]](
// HOST: define internal fastcc void @[[HELPER]](
// HOST: call void @__cilkrts_enter_frame(
// HOST: %[[STREAM:[^ ]+]] = call ptr @__kitcuda_launch_kernel(
// HOST: call void @__kitcuda_sync_thread_stream(ptr %[[STREAM]])

// The cloned CPU loop retains its loop metadata and source locations.
// HOST: br i1 {{.*}}, !dbg ![[LOC:[0-9]+]], !llvm.loop ![[LOOP:[0-9]+]]
// HOST-DAG: ![[LOC]] = !DILocation(
// HOST-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[LOC]],

// DEVICE-LABEL: define {{.*}} @"_cuabi_kern_{{[^" ]+}}"(
// DEVICE: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// DEVICE: store i32
// PTX: .entry _cuabi_kern_
