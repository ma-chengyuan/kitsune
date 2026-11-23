// RUN: %clang_cc1 %s -triple amd64-freebsd -fopencilk -verify -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 %s -x c++ -fopencilk -verify -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// expected-no-diagnostics

void identity_long(void *v);
void reduce_long(void *l, void *r);

// CHECK: @global = {{.*}}global i64 1
// CHECK-LABEL: cxx_global_var_init
// CHECK-NOT: store i64 1, ptr @global
// CHECK: call void @llvm.reducer.register.i64(ptr @global, i64 8,
// CHECK-NOT: store i64 1, ptr @global
// CHECK: ret void
long _Hyperobject(identity_long, reduce_long) global = 1;

// CHECK: call void @llvm.reducer.unregister
