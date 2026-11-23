// RUN: %clang -fcilkplus -fsyntax-only %s
// RUN: %clang -fcilkplus -x c-header %s -o %t.pch
// RUN: %clang -fcilkplus -S -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=IR < %t.ll
// RUN: %clang -fcilkplus -ftapir=none -S -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang -fcilkplus -ftapir=serial -c %s -o %t.o
// RUN: %clang -fcilkplus -ftapir=serial -c %t.ll -o %t.ir.o
// RUN: not %clang -fcilkplus -S %s -o %t.s 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang -fcilkplus -c %s -o %t.o 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang -fcilkplus -c %t.ll -o %t.ir.o 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang -fcilkplus -ftapir=none -c %s -o %t.none.o 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang -fcilkplus %s -o %t 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang -fcilkplus %t.o -o %t 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: %clang -fcilkplus -flto -c %s -o %t.lto.o
// RUN: llvm-dis %t.lto.o -o - | FileCheck %s --check-prefix=IR
// RUN: not %clang -fcilkplus -flto %t.lto.o -o %t 2>&1 | FileCheck %s --check-prefix=NATIVE
// RUN: not %clang --target=x86_64-unknown-linux-gnu -fcilkplus -flto -ffat-lto-objects \
// RUN:   -c %s -o %t.fat.o 2>&1 | FileCheck %s --check-prefix=NATIVE

// IR: detach within
// IR: sync within
// NATIVE: error: native compilation and linking with '-fcilkplus' require
// NATIVE-SAME: '-ftapir='
// NATIVE-SAME: '-fopencilk' for the OpenCilk runtime

extern void work(void);
int main(void) {
  _Cilk_spawn work();
  _Cilk_sync;
  return 0;
}
