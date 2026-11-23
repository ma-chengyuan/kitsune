// RUN: c-index-test -cursor-at=%s:15:22 -cursor-at=%s:15:35 -cursor-at=%s:17:10 -cursor-at=%s:17:28 -cursor-at=%s:17:38 %s -fopencilk -ftapir=none | FileCheck %s
// CHECK: DeclRefExpr=start:
// CHECK: DeclRefExpr=bound:
// CHECK: TypeRef=struct Item:
// CHECK: DeclRefExpr=identity:
// CHECK: DeclRefExpr=reduce:

struct Item { int n; };
void identity(void *);
void reduce(void *, void *);
int start(), bound();
void use(int);

void run() {
  _Cilk_for (int i = start(); i < bound(); ++i) use(i);
  struct Item cilk_reducer(identity, reduce) first;
  struct Item cilk_reducer(identity, reduce) second;
}

// The two declarations share a semantic type but spell their callbacks at
// different locations. Preserve both uses through serialization.
// RUN: c-index-test -write-pch %t.ast -x c++ -fopencilk -ftapir=none %s
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck %s --check-prefix=PCH
// PCH: :16:28: DeclRefExpr=identity:
// PCH: :16:38: DeclRefExpr=reduce:
// PCH: :17:28: DeclRefExpr=identity:
// PCH: :17:38: DeclRefExpr=reduce:
