// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -Wno-source-uses-cilk-plus -include %s -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -Wno-source-uses-cilk-plus -emit-pch %s -o %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -Wno-source-uses-cilk-plus -include-pch %t -emit-llvm %s -o - | FileCheck %s

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

template <int Grain> void counted(int *a, int n) {
  [[tapir::grain_size(Grain)]]
  _Cilk_for (int i = 0; i < n; ++i)
    a[i] = i;
}

template <class T, int Grain> void ranged(T (&a)[8]) {
  [[tapir::grain_size(Grain + 1)]]
  _Cilk_for (T &value : a)
    ++value;
}

#else

template void counted<4>(int *, int);
template void ranged<int, 6>(int (&)[8]);

#endif

// CHECK-LABEL: define {{.*}}void @_Z7countedILi4EEvPii(
// CHECK: br i1 {{.*}}, !llvm.loop [[COUNTED:![0-9]+]]
// CHECK-LABEL: define {{.*}}void @_Z6rangedIiLi6EEvRA8_T_(
// CHECK: br i1 {{.*}}, !llvm.loop [[RANGED:![0-9]+]]
// CHECK: [[COUNTED]] = distinct !{[[COUNTED]], {{.*}}, [[COUNTED_GRAIN:![0-9]+]], {{.*}}}
// CHECK: [[COUNTED_GRAIN]] = !{!"tapir.loop.grainsize", i32 4}
// CHECK: [[RANGED]] = distinct !{[[RANGED]], {{.*}}, [[RANGED_GRAIN:![0-9]+]], {{.*}}}
// CHECK: [[RANGED_GRAIN]] = !{!"tapir.loop.grainsize", i32 7}
