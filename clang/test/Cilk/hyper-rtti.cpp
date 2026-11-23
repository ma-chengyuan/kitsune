// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fopencilk -ftapir=none -verify -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

namespace std { class type_info; }

using IntHyperobjectPtr = int _Hyperobject *;

const std::type_info &int_hyperobject_pointer() {
  return typeid(IntHyperobjectPtr);
}

struct Base {};
struct Derived : Base {};
using ClassHyperobjectPtr = Derived _Hyperobject *;

const std::type_info &class_hyperobject_pointer() {
  return typeid(ClassHyperobjectPtr);
}

// CHECK-DAG: @_ZTIHi = {{.*}}constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSHi }
// CHECK-DAG: @_ZTIPHi = {{.*}}constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSPHi, i32 0, ptr @_ZTIHi }
// CHECK-DAG: @_ZTIH7Derived = {{.*}}constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv123__fundamental_type_infoE, i64 2), ptr @_ZTSH7Derived }
// CHECK-DAG: @_ZTIPH7Derived = {{.*}}constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSPH7Derived, i32 0, ptr @_ZTIH7Derived }

// CHECK-LABEL: define {{.*}}@_Z23int_hyperobject_pointerv(
// CHECK: ret ptr @_ZTIPHi
// CHECK-LABEL: define {{.*}}@_Z25class_hyperobject_pointerv(
// CHECK: ret ptr @_ZTIPH7Derived
