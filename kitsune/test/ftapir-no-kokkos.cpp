// Check that the correct error is emitted if the -fkokkos or -fkokkos-no-init
// flags are specified but Kokkos support was not enabled.
//
// REQUIRES: kitsune-no-kokkos
//
// RUN: not %kitxx -fsyntax-only -fkokkos %s 2>&1 | FileCheck %s
// RUN: not %kitxx -fsyntax-only -fkokkos-no-init %s 2>&1 | FileCheck %s
//
// CHECK: error: kokkos support was not enabled when kitsune was built
