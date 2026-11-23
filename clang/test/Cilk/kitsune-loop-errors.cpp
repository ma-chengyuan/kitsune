// RUN: %clang_cc1 -std=c++17 -fopencilk -fopencilk-kitsune -fsyntax-only -verify %s

void invalid_targets(int *a, int n) {
  [[tapir::target("unknown")]] // expected-error {{unknown tapir target}}
  _Cilk_for (int i = 0; i < n; ++i)
    a[i] = i;

  [[tapir::deferred_sync]] // expected-error {{'deferred_sync' attribute only applies to 'parallel for' statements}}
  for (int i = 0; i < n; ++i)
    a[i] = i;
}
