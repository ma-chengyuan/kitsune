// RUN: c-index-test -test-print-type %s -target x86_64-unknown-linux-gnu -fopencilk -ftapir=none -Wno-source-uses-cilk-plus | FileCheck %s

void identity(void *);
void reduce(void *, void *);
int work();

void update(int n) {
  int cilk_reducer(identity, reduce) reducer;
  _Cilk_scope {
    _Cilk_spawn work();
    int value = _Cilk_spawn work();
    _Cilk_sync;
    _Cilk_for (int i = 0; i < n; ++i)
      reducer += i;
    int values[] = {1, 2};
    _Cilk_for (int i : values)
      reducer += i;
  }
}

// CHECK: VarDecl=reducer:{{.*}} [typekind=Hyperobject]
// CHECK: CilkScopeStmt=
// CHECK: CilkSpawnStmt=
// CHECK: CilkSpawnExpr={{.*}} [type=int] [typekind=Int]
// CHECK: CilkSyncStmt=
// CHECK: CilkForStmt=
// CHECK: CilkForRangeStmt=
