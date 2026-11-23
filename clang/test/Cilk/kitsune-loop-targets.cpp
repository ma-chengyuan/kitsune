// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopencilk -fopencilk-kitsune -ftapir=none -emit-llvm %s -o - | FileCheck %s

// The hybrid driver hands these loops to Kitsune before either is outlined.
extern "C" void update(int *a, int n) {
  _Cilk_for (int i = 0; i < n; ++i)
    a[i] = i;
  [[tapir::target("cuda"), tapir::grain_size(8), tapir::deferred_sync]]
  _Cilk_for (int i = 0; i < n; ++i)
    a[i] += 1;
}

// CHECK-LABEL: define {{.*}}void @update(
// CHECK: detach within
// CHECK: br i1 {{.*}}, !llvm.loop [[CPU:![0-9]+]]
// CHECK: detach within
// CHECK: br i1 {{.*}}, !llvm.loop [[GPU:![0-9]+]]
// CHECK: [[CPU]] = distinct !{[[CPU]], [[PROGRESS:![0-9]+]], [[SPAWN:![0-9]+]], [[OPENCILK:![0-9]+]]}
// CHECK: [[OPENCILK]] = !{!"tapir.loop.target", i32 6}
// CHECK: [[GPU]] = distinct !{[[GPU]], [[PROGRESS]], [[SPAWN]], [[GRAIN:![0-9]+]], [[CUDA:![0-9]+]], [[DEFER:![0-9]+]]}
// CHECK-DAG: [[GRAIN]] = !{!"tapir.loop.grainsize", i32 8}
// CHECK-DAG: [[CUDA]] = !{!"tapir.loop.target", i32 2}
// CHECK-DAG: [[DEFER]] = !{!"tapir.loop.deferred_sync", i1 true}
