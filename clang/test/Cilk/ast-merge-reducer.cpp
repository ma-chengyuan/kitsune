// RUN: %clang_cc1 -std=c++17 -fopencilk -emit-pch -x c++-header %s -o %t.pch
// RUN: %clang_cc1 -std=c++17 -fopencilk -ast-merge %t.pch -ast-dump -x c++ /dev/null | FileCheck %s

void identity(void *);
void reduce(void *, void *);
void update() {
  int cilk_reducer(identity, reduce) total = 0;
  ++total;
}

// CHECK: FunctionDecl {{.*}} update
// CHECK: VarDecl {{.*}} total
// CHECK: UnaryOperator {{.*}} prefix '++'
