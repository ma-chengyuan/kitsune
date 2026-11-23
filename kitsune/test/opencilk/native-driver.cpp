// REQUIRES: kitsune-opencilk, system-linux
// RUN: %clangxx -### -fopencilk %s 2>&1 | FileCheck %s --check-prefix=CPU \
// RUN:   --implicit-check-not=-lkitrt --implicit-check-not=-lcuda
// RUN: %clangxx -### -fopencilk -fopencilk-kitsune -ftapir=none \
// RUN:   -S -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=HANDOFF \
// RUN:   --implicit-check-not=opencilk-abi-bitcode --implicit-check-not=libopencilk
// RUN: %clangxx -### -fopencilk -static %s 2>&1 | FileCheck %s --check-prefix=STATIC
// RUN: not %clangxx -fopencilk -static-pie %s -o %t 2>&1 | FileCheck %s \
// RUN:   --check-prefix=STATIC-PIE --implicit-check-not='linker command failed'
// RUN: %clangxx -fopencilk -static-pie -c %s -o %t.o
// RUN: not %clangxx -fopencilk -static-pie %t.o -o %t 2>&1 | FileCheck %s \
// RUN:   --check-prefix=STATIC-PIE --implicit-check-not='linker command failed'
// RUN: not %clangxx -fopencilk -ftapir=opencilk -static-pie %t.o -o %t 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATIC-PIE
// RUN: %clangxx -### -fopencilk -static-pie -nostdlib %t.o 2>&1 | FileCheck %s \
// RUN:   --check-prefix=MANUAL-STATIC-PIE --implicit-check-not=libopencilk --implicit-check-not='error:'

// CPU: "-cc1"
// CPU-SAME: "-fopencilk"
// CPU-SAME: "--opencilk-abi-bitcode=[[CILK_LIB:[^"]+]]/libopencilk-abi{{[^"/]*}}.bc"
// CPU: "[[CILK_LIB]]/libopencilk-personality-cpp{{[^"/]*}}.so"
// CPU-SAME: "[[CILK_LIB]]/libopencilk{{[^"/]*}}.so"
// CPU-SAME: "-rpath" "[[CILK_LIB]]"

// HANDOFF: "-cc1"
// HANDOFF-SAME: "-emit-llvm"
// HANDOFF-SAME: "-ftapir=none"
// HANDOFF-SAME: "-fopencilk-kitsune"

// STATIC: "{{[^"]+}}/libopencilk-personality-cpp{{[^"/]*}}.a"
// STATIC-SAME: "{{[^"]+}}/libopencilk{{[^"/]*}}.a"
// STATIC-PIE: error: the OpenCilk runtime does not support linking with '-static-pie'
// STATIC-PIE-SAME: '-static' for a static executable
// MANUAL-STATIC-PIE: "-static" "-pie"

int main() { return 0; }
