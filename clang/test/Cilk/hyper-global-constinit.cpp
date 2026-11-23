// RUN: %clang_cc1 %s -std=c++20 -triple x86_64-unknown-linux-gnu -fopencilk -ftapir=none -verify -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// expected-no-diagnostics

void identity(void *);
void reduce(void *, void *);

constexpr int initial_value() {
  return __builtin_is_constant_evaluated() ? 42 : 99;
}

constinit int _Hyperobject(identity, reduce) global = initial_value();

// CHECK: @global = {{.*}}global i32 42
// CHECK-LABEL: define internal void @__cxx_global_var_init()
// CHECK-NOT: call {{.*}}@_Z13initial_valuev(
// CHECK-NOT: store {{.*}}ptr @global
// CHECK: call void @llvm.reducer.register.i64(ptr @global, i64 4, ptr @_Z8identityPv, ptr @_Z6reducePvS_)
// CHECK-NOT: call {{.*}}@_Z13initial_valuev(
// CHECK-NOT: store {{.*}}ptr @global
// CHECK: ret void
// CHECK-LABEL: define internal void @__cxx_global_hyperobject_dtor(
// CHECK: call void @llvm.reducer.unregister(ptr @global)
