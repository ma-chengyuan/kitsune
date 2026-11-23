//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
std::size_t LastSize;
unsigned Frees;
} // namespace

extern "C" void *__kitcuda_mem_alloc_managed(std::size_t Size) {
  LastSize = Size;
  return std::malloc(Size);
}

extern "C" void __kitcuda_mem_free(void *Pointer) {
  ++Frees;
  std::free(Pointer);
}

int main() {
  void *Scalar = ::operator new(32);
  if (LastSize != 32)
    return 1;
  ::operator delete(Scalar, std::size_t{32});
  if (Frees != 1)
    return 2;

  void *Array = ::operator new[](48, std::nothrow);
  if (!Array || LastSize != 48)
    return 3;
  ::operator delete[](Array);
  if (Frees != 2)
    return 4;

  // Keep the libc calls observable in optimized test builds.
  auto *volatile Malloc = std::malloc;
  auto *volatile Free = std::free;
  void *CAllocation = Malloc(16);
  if (!CAllocation)
    return 5;
  Free(CAllocation);
  if (LastSize != 48 || Frees != 2)
    return 6;
}
