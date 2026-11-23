// RUN: %clang_cc1 -std=c++17 -fopencilk -fsyntax-only -verify %s

template <bool Break> void iterate(int (&array)[8]) {
  // The range and its iterator types are known before template substitution.
  // The break is only visible to Sema once the chosen arm is known.
  _Cilk_for (int entry : array) // expected-warning {{'cilk_for' support for for-range loops is currently experimental}} expected-error {{cannot break from a 'cilk_for' loop}}
    __builtin_choose_expr(Break, ({ break; }), (void)entry);
}

template void iterate<false>(int (&)[8]);
template void iterate<true>(int (&)[8]); // expected-note {{in instantiation of function template specialization 'iterate<true>' requested here}}
