// REQUIRES: kitsune-opencilk, system-linux
// RUN: %clangxx -### -fopencilk %s 2>&1 | FileCheck %s --check-prefix=CPU \
// RUN:   --implicit-check-not=-lkitrt --implicit-check-not=-lcuda
// RUN: %clangxx -### -fopencilk -fopencilk-kitsune -ftapir=none \
// RUN:   -S -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=HANDOFF \
// RUN:   --implicit-check-not=opencilk-abi-bitcode --implicit-check-not=libopencilk

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

int main() { return 0; }
