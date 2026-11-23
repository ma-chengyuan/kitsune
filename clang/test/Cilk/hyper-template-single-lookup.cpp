// RUN: %clang_cc1 %s -fopencilk -emit-llvm -disable-llvm-passes -o - | FileCheck %s

struct Sum {
  void operator+=(int);
};
void init(void *);
void merge(void *, void *);

template <typename T> void accumulate(T n) {
  Sum cilk_reducer(init, merge) sum;
  _Cilk_for (int i = 0; i < n; ++i)
    sum += i;
}
template void accumulate(int);

// A stolen iteration must use the view returned by the first lookup.
// CHECK-LABEL: define {{.*}}accumulate
// CHECK: %[[VIEW:[^ ]+]] = call ptr @llvm.hyper.lookup.i64(ptr %sum,
// CHECK-NOT: call ptr @llvm.hyper.lookup
// CHECK: {{call|invoke}} void @_ZN3SumpLEi(ptr {{[^,]*}}%[[VIEW]],
