// UNSUPPORTED: zera
// RUN: %clang --target=x86_64-unknown-linux-gnu -fzera -fsyntax-only \
// RUN:   -resource-dir %t/absent %s
// RUN: not %clang --target=x86_64-unknown-linux-gnu -### -fzera -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNAVAILABLE

// UNAVAILABLE: error: '-fzera' requires a build with OpenCilk and CUDA Tapir support and the NVPTX target

static void work(void) {}

int main(void) {
  _Cilk_spawn work();
  _Cilk_sync;
  return 0;
}
