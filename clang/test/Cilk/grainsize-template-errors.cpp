// RUN: %clang_cc1 -std=c++17 -fopencilk -Wno-source-uses-cilk-plus -fsyntax-only -verify %s

template <int Grain> void counted(int *a, int n) {
  [[tapir::grain_size(Grain)]] // expected-error {{invalid value '0'; must be positive}}
  _Cilk_for (int i = 0; i < n; ++i)
    a[i] = i;
}
template void counted<0>(int *, int); // expected-note {{in instantiation of function template specialization 'counted<0>' requested here}}

template <class T, auto Grain> void ranged(T (&a)[8]) {
  [[tapir::grain_size(Grain)]] // expected-error {{invalid argument of type 'bool'; expected an integer type}}
  _Cilk_for (T &value : a)
    ++value;
}
template void ranged<int, true>(int (&)[8]); // expected-note {{in instantiation of function template specialization 'ranged<int, true>' requested here}}
