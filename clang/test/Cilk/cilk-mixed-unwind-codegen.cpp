// Check that Clang may generate functions calls that can throw with or without
// a landingpad in the same Cilk scope.
//
// RUN: %clang_cc1 -fopencilk -fcxx-exceptions -fexceptions -ftapir=none -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

int bar(int n);
void foo(int n) {
    cilk_for (int i = 0; i < n; ++i) {
        int w = bar(i);
        throw bar(w);
    }
}

// CHECK-LABEL: define {{.*}}void @_Z3fooi(i32 {{.*}}%n)

// Check for detach with an unwind destination
// CHECK: detach within %[[SYNCREG:.+]], label %[[PFOR_BODY_ENTRY:.+]], label %[[PFOR_INC:.+]] unwind label %[[DETACH_LPAD:.+]]

// CHECK: [[PFOR_BODY_ENTRY]]:

// Check for call to function bar that might throw.
// CHECK: call {{.*}}i32 @_Z3bari(i32

// Check for invoke of function bar
// CHECK: invoke noundef i32 @_Z3bari(i32
// CHECK-NEXT: to label %[[INVOKE_CONT:.+]] unwind label %[[TASK_LPAD:.+]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: call void @__cxa_throw(ptr
// CHECK-NEXT: unreachable

// CHECK: [[TASK_LPAD]]:
// CHECK-NEXT: landingpad
// CHECK-NEXT: cleanup
// CHECK: invoke void @llvm.detached.rethrow.sl_p0i32s(token %[[SYNCREG]], { ptr, i32 } %{{.*}})
// CHECK-NEXT: to label %[[UNREACHABLE:.+]] unwind label %[[DETACH_LPAD]]
