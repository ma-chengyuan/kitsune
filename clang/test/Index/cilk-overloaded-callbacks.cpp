// RUN: c-index-test -cursor-at=%s:17:32 -cursor-at=%s:17:56 -cursor-at=%s:18:32 -cursor-at=%s:18:56 %s -fopencilk -ftapir=none | FileCheck %s --check-prefix=CURSOR
// RUN: c-index-test -write-pch %t.ast -x c++ -fopencilk -ftapir=none %s
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck %s --check-prefix=PCH --implicit-check-not=OverloadedDeclRef

// CURSOR: DeclRefExpr=identity:12:6
// CURSOR-NEXT: DeclRefExpr=reduce:13:6
// CURSOR-NEXT: DeclRefExpr=identity:12:6
// CURSOR-NEXT: DeclRefExpr=reduce:13:6

namespace callbacks {
void identity(int *);
void identity(void *);
void reduce(void *, void *);
void reduce(int *, int *);
}
void run() {
  int cilk_reducer((callbacks::identity), (&callbacks::reduce)) first;
  int cilk_reducer((callbacks::identity), (&callbacks::reduce)) second;
}

// PCH: :17:20: ParenExpr=
// PCH: :17:32: DeclRefExpr=identity:12:6
// PCH: :17:21: NamespaceRef=callbacks:10:11
// PCH: :17:43: ParenExpr=
// PCH: :17:44: UnaryOperator=
// PCH: :17:56: DeclRefExpr=reduce:13:6
// PCH: :17:45: NamespaceRef=callbacks:10:11
// PCH: :18:20: ParenExpr=
// PCH: :18:32: DeclRefExpr=identity:12:6
// PCH: :18:43: ParenExpr=
// PCH: :18:44: UnaryOperator=
// PCH: :18:56: DeclRefExpr=reduce:13:6
