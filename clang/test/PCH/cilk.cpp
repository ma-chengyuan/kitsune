// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -emit-pch -x c++-header %s -o %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fopencilk -ftapir=none -include-pch %t -ast-dump-all -ast-dump-filter=update_classic %s | FileCheck %s --check-prefix=AST

#ifndef HEADER
#define HEADER

int compute(int);

template <class T> void update_classic(T *data, int n) {
  int value = _Cilk_spawn compute(n);
  _Cilk_sync;
  _Cilk_for (int i = 0; i < n; ++i)
    data[i] = value;
}

inline void update_range(int (&data)[4]) {
  _Cilk_scope {
    _Cilk_for (int &value : data)
      value = compute(value);
  }
}

#else

void use_pch(int (&data)[4]) {
  update_classic(data, 4);
  update_range(data);
}

#endif

// The transformed loop tests and increments hidden variables. The original
// test and increment still need to refer to the user's loop variable.
// AST-LABEL: FunctionTemplateDecl
// AST: CilkForStmt
// AST: VarDecl [[INDEX:0x[0-9a-f]+]] {{.*}} i 'int'
// AST: BinaryOperator {{.*}} '<'
// AST-NEXT: {{.*}}ImplicitCastExpr {{.*}} <LValueToRValue>
// AST-NEXT: {{.*}}DeclRefExpr {{.*}} Var [[INDEX]] 'i'
// AST: UnaryOperator {{.*}} prefix '++'
// AST-NEXT: {{.*}}DeclRefExpr {{.*}} Var [[INDEX]] 'i'

// IR-LABEL: define {{.*}} @_Z14update_classicIiEvPT_i(
// IR: detach within
// IR: sync within
// IR: detach within

// IR-LABEL: define {{.*}} @_Z12update_rangeRA4_i(
// IR: pfor.cond:
// IR: detach within
// IR: reattach within
