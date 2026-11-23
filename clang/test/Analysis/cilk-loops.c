// RUN: %clang --analyze -x c -fopencilk -Xclang -verify %s -o %t
// RUN: %clang --analyze -x c++ -std=c++17 -fopencilk -Wno-source-uses-cilk-plus -Xclang -verify %s -o %t

void use(int);

void counted(void) {
  _Cilk_for (int i = 0; i < 6; ++i)
    use(i);
}

void counted_body(void) {
  _Cilk_for (int i = 0; i < 2; ++i) {
    if (i == 0)
      continue;
    *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
  }
}

void counted_zero_trip(void) {
  _Cilk_for (unsigned i = 3; i < 2; ++i)
    *(volatile int *)0 = 1;
}

#ifdef __cplusplus
void ranged(void) {
  int values[] = {1, 2, 3};
  _Cilk_for (int value : values)
    use(value);
}

void ranged_body(void) {
  int values[] = {1, 2, 3};
  _Cilk_for (int value : values)
    if (value == 2)
      *(volatile int *)0 = 1; // expected-warning {{Dereference of null pointer}}
}
#endif
