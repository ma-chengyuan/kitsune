// REQUIRES: x86-registered-target, zera
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzera -O1 -funroll-loops -vectorize-loops -vectorize-slp -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s | FileCheck %s --check-prefix=PIPE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzera -O2 -funroll-loops -vectorize-loops -vectorize-slp -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s | FileCheck %s --check-prefix=PIPE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzera -O3 -funroll-loops -vectorize-loops -vectorize-slp -emit-llvm -o /dev/null -mllvm -print-pipeline-passes %s | FileCheck %s --check-prefix=PIPE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzera -O2 -debug-info-kind=standalone --opencilk-abi-bitcode=%S/Inputs/libopencilk-abi.bc -emit-llvm -o - %s | FileCheck %s --check-prefix=MIXED --implicit-check-not='detach within' --implicit-check-not='sync within'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopencilk -O2 --opencilk-abi-bitcode=%S/Inputs/libopencilk-abi.bc -emit-llvm -o - %s | FileCheck %s --check-prefix=CPU --implicit-check-not=__kitcuda_memcpy
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fzera -O0 -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=LEVEL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -std=c++17 -fzera -O2 -emit-llvm -o /dev/null -DBAD_TARGET -verify %s

// PIPE: loop-vectorize<interleave-forced-only;vectorize-forced-only;>
// PIPE-NOT: slp-vectorizer
// PIPE-SAME: loop-spawning
// PIPE-SAME: tapir2target
// PIPE-SAME: tapir-memops-replacement
// PIPE-SAME: loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>
// PIPE-SAME: slp-vectorizer
// LEVEL: error: '-fzera' supports optimization levels -O1, -O2 and -O3 (got '-O0')

void work(int);

// MIXED-LABEL: define {{.*}}void @parallel(
// MIXED: call {{.*}}@parallel.outline_{{[^ (]+}}(
void parallel(int n) {
  _Cilk_for (int i = 0; i < n; ++i)
    work(i);
}

// MIXED-LABEL: define {{.*}}void @copy(
// MIXED: call void @__kitcuda_memcpy(
// CPU-LABEL: define {{.*}}void @copy(
// CPU: call void @llvm.memcpy
void copy(void *dst, const void *src, unsigned long n) {
  __builtin_memcpy(dst, src, n);
}

// MIXED-DAG: !DISubprogram(name: "parallel"
// MIXED-DAG: !{!"llvm.loop.unroll.disable"}

#ifdef BAD_TARGET
void unsupported(int n) {
  [[tapir::target("hip")]] // expected-error {{unsupported Tapir loop target 'hip' with '-fzera'}}
  _Cilk_for (int i = 0; i < n; ++i)
    work(i);
}
#endif
