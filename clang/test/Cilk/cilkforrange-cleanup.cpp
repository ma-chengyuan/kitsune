// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-unknown-linux-gnu -fopencilk -ftapir=none -verify -emit-llvm -disable-llvm-passes -o - | FileCheck %s

struct Range {
  Range(int);
  ~Range();
  int *begin();
  int *end();
};

void use(int);

void iterate(int n) {
  _Cilk_for (int entry : Range(n)) // expected-warning {{'cilk_for' support for for-range loops is currently experimental}}
    use(entry);
}

// CHECK-LABEL: define {{.*}}void @_Z7iteratei(
// CHECK: pfor.initial.entry:
// CHECK: br i1 {{.*}}, label %pfor.cond, label %pfor.initial.cleanup
// CHECK: pfor.initial.cleanup:
// CHECK-NOT: sync within
// CHECK: br label %[[CLEANUP:[a-zA-Z0-9._]+]]
// CHECK: pfor.cond.cleanup:
// CHECK: sync within %{{.*}}, label %[[SYNC_CONT:[a-zA-Z0-9._]+]]
// CHECK: [[SYNC_CONT]]:
// CHECK-NEXT: br label %[[CLEANUP]]
// CHECK: [[CLEANUP]]:
// CHECK: call void @_ZN5RangeD1Ev(
// CHECK-NEXT: br label %pfor.end
