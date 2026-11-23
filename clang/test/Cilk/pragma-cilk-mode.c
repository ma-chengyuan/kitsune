// RUN: %clang_cc1 -x c -fsyntax-only -Wunknown-pragmas -verify=off %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wunknown-pragmas -verify=off %s
// RUN: %clang_cc1 -x c -fsyntax-only -fopencilk -verify=on %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fopencilk -verify=on %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fcilkplus -verify=on %s

void f(int *a) {
  // off-warning@+1 {{unknown pragma ignored}}
#pragma cilk grainsize 2
  for (int i = 0; i < 8; ++i) // on-error {{expected a Cilk keyword to follow '#pragma cilk'}}
    a[i] = i;
}
