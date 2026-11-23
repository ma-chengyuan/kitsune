// REQUIRES: zera
// RUN: rm -rf %t
// RUN: mkdir -p %t/relocated/lib/x86_64-unknown-linux-gnu %t/cuda/bin %t/cuda/nvvm/libdevice
// RUN: touch %t/relocated/lib/x86_64-unknown-linux-gnu/libopencilk-abi.bc
// RUN: touch %t/relocated/lib/x86_64-unknown-linux-gnu/libopencilk.so
// RUN: touch %t/relocated/lib/x86_64-unknown-linux-gnu/libopencilk-personality-cpp.so
// RUN: touch %t/relocated/lib/kitcuda.bc %t/relocated/lib/libkitrt.so %t/relocated/lib/libzera_alloc.a
// RUN: touch %t/cuda/bin/ptxas %t/cuda/bin/fatbinary %t/cuda/nvvm/libdevice/libdevice.10.bc
// RUN: touch %t/existing.o

// RUN: env CUDAABI_TARGET=sm_00 %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c \
// RUN:   -resource-dir %t/relocated --cuda-path=%t/cuda \
// RUN:   --cuabi-arch=sm_80 --cuabi-dont-fuse-reducers --cuabi-naive-atomics %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CODEGEN --implicit-check-not='error:'
// CODEGEN: "-cc1"
// CODEGEN-SAME: "-emit-obj"
// CODEGEN-SAME: "-O1"
// CODEGEN-SAME: "-fzera"
// CODEGEN-SAME: "--opencilk-abi-bitcode={{[^"]*}}/relocated/lib/x86_64-unknown-linux-gnu/libopencilk-abi.bc"
// CODEGEN-SAME: "--kitcuda-abi-bitcode={{[^"]*}}/relocated/lib/kitcuda.bc"
// CODEGEN-SAME: "--kitcuda-libdevice={{[^"]*}}/cuda/nvvm/libdevice/libdevice.10.bc"
// CODEGEN-SAME: "--kitcuda-ptxas={{[^"]*}}/cuda/bin/ptxas"
// CODEGEN-SAME: "--kitcuda-fatbinary={{[^"]*}}/cuda/bin/fatbinary"
// CODEGEN-SAME: "-mllvm" "--cuabi-arch=sm_80"
// CODEGEN-SAME: "-mllvm" "--cuabi-dont-fuse-reducers"
// CODEGEN-SAME: "-mllvm" "--cuabi-naive-atomics"
// CODEGEN-NOT: libzera_alloc

// RUN: env CUDAABI_TARGET=sm_00 %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c \
// RUN:   -resource-dir %t/relocated --cuda-path=%t/cuda \
// RUN:   -mllvm -cuabi-arch -mllvm sm_80 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LLVM-ARCH --implicit-check-not='error:'
// LLVM-ARCH: "-mllvm" "-cuabi-arch" "-mllvm" "sm_80"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -O2 -g \
// RUN:   -resource-dir %t/relocated --cuda-path=%t/cuda %s %t/existing.o 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=SOURCE,LINK --implicit-check-not='error:'
// SOURCE: "-cc1"
// SOURCE-SAME: "-debug-info-kind=constructor"
// SOURCE-SAME: "-O2"
// LINK: "{{[^"]*}}/existing.o"
// LINK-SAME: "[[LIBDIR:[^"]*]]/x86_64-unknown-linux-gnu/libopencilk-personality-cpp.so"
// LINK-SAME: "[[LIBDIR]]/x86_64-unknown-linux-gnu/libopencilk.so"
// LINK-SAME: "-rpath" "[[LIBDIR]]/x86_64-unknown-linux-gnu"
// LINK-SAME: "--push-state" "--whole-archive" "[[LIBDIR]]/libzera_alloc.a" "--pop-state"
// LINK-SAME: "[[LIBDIR]]/libkitrt.so"
// LINK-SAME: "-L" "[[CUDA:[^"]*]]/cuda/lib64"
// LINK-SAME: "--push-state" "--no-as-needed" "-lcudart" "--pop-state"
// LINK-SAME: "-rpath" "[[LIBDIR]]" "-rpath" "[[CUDA]]/cuda/lib64"

// RUN: rm %t/relocated/lib/kitcuda.bc
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -O0 \
// RUN:   -resource-dir %t/relocated --cuda-path=%t/cuda -Wl,--as-needed %t/existing.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not='error:' --implicit-check-not=-cc1

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fzera -E -dM \
// RUN:   -resource-dir %t/absent %s | FileCheck %s --check-prefix=CILK
// CILK: #define __cilk 301

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -E -O0 \
// RUN:   -resource-dir %t/absent --cuda-path=%t/absent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PREPROCESS --implicit-check-not='warning:' \
// RUN:   --implicit-check-not='error:' --implicit-check-not=--opencilk-abi-bitcode \
// RUN:   --implicit-check-not=--kitcuda-abi-bitcode --implicit-check-not=libzera_alloc
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -fsyntax-only \
// RUN:   -resource-dir %t/absent %s 2>&1 | FileCheck %s --check-prefix=SYNTAX \
// RUN:   --implicit-check-not='error:' --implicit-check-not=--opencilk-abi-bitcode \
// RUN:   --implicit-check-not=--kitcuda-abi-bitcode --implicit-check-not=libzera_alloc
// PREPROCESS: "-cc1"
// PREPROCESS-SAME: "-E"
// PREPROCESS-SAME: "-O0"
// PREPROCESS-SAME: "-fzera"
// SYNTAX: "-cc1"
// SYNTAX-SAME: "-fsyntax-only"
// SYNTAX-SAME: "-fzera"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -### -fopencilk \
// RUN:   -resource-dir %t/relocated %s 2>&1 | FileCheck %s --check-prefix=CPU \
// RUN:   --implicit-check-not='error:' --implicit-check-not=--kitcuda-abi-bitcode \
// RUN:   --implicit-check-not=libkitrt --implicit-check-not=libzera_alloc
// CPU: "-cc1"
// CPU-SAME: "-fopencilk"
// CPU-SAME: "--opencilk-abi-bitcode={{[^"]*}}/relocated/lib/x86_64-unknown-linux-gnu/libopencilk-abi.bc"
// CPU: /relocated/lib/x86_64-unknown-linux-gnu/libopencilk-personality-cpp.so"

// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c -O0 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-OPT
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c -flto=thin %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-LTO
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c -ftapir=cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-TAPIR
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -c --cuabi-arch=sm_80 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-CUABI
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c --cuabi-arch=sm_00 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-ARCH
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c -mllvm -cuabi-arch=sm_00 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-ARCH
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c -mllvm -cuabi-arch -mllvm sm_00 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-ARCH
// RUN: not env CUDAABI_TARGET=sm_00 %clangxx --target=x86_64-unknown-linux-gnu -### -fzera -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-ARCH-ENV
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -### -fzera \
// RUN:   -resource-dir %t/relocated --cuda-path=%t/cuda -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING-KITCUDA
// BAD-OPT: error: '-fzera' supports optimization levels -O1, -O2 and -O3 (got '-O0')
// BAD-LTO: error: invalid argument '-flto=thin' not allowed with '-fzera'
// BAD-TAPIR: error: invalid argument '-ftapir=cuda' not allowed with '-fzera'
// BAD-CUABI: error: invalid argument '--cuabi-arch=sm_80' only allowed with '-fzera'
// BAD-ARCH: error: unsupported CUDA architecture 'sm_00' for '-fzera'
// BAD-ARCH-ENV: error: unsupported CUDA architecture 'sm_00' in CUDAABI_TARGET for '-fzera'
// MISSING-KITCUDA: error: cannot find '-fzera' KitCUDA ABI bitcode: '{{[^']*}}/relocated/lib/kitcuda.bc'

int main() {}
