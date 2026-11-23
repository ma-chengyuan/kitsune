#include <cilk/cilk.h>

[[gnu::noinline]] static void apply(int *out, int count) {
  [[tapir::target("cuda"), tapir::grain_size(1)]]
  cilk_for (int i = 0; i < count; ++i)
    out[i] = i + 29;
}

void update_b(int *out, int count) { apply(out, count); }
