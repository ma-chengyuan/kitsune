//===- LoopSpawningTI.h - Spawn loop iterations efficiently -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass modifies Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"
#include <optional>

namespace llvm {

/// The LoopSpawning Pass.
class LoopSpawningPass : public PassInfoMixin<LoopSpawningPass> {
  std::optional<CudaABIOptions> CudaOptions;

public:
  LoopSpawningPass() = default;
  explicit LoopSpawningPass(const CudaABIOptions *Options) {
    if (Options)
      CudaOptions = *Options;
  }

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
