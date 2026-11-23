//===--- Tapir.h - C Language Family Language Options -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines helper functions for processing flags related to Tapir.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TAPIR_H
#define LLVM_CLANG_DRIVER_TAPIR_H

#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

namespace llvm {
namespace opt {
class ArgList;
}
} // namespace llvm

namespace clang {

/// Return the explicit -ftapir target, or OpenCilk by default with -fopencilk
/// or -fzera. Return std::nullopt if no valid target is selected.
std::optional<llvm::TapirTargetID>
parseTapirTarget(const llvm::opt::ArgList &Args);

/// Parse the -ftapir-nvarch flag if it is present in the arguments. If the flag
/// is not present, return std::nullopt. If the flag is present and its value
/// is not valid, return Last_TapirNVArchTargetID.
std ::optional<llvm::TapirNVArchTargetID>
parseTapirNVArchTarget(const llvm::opt::ArgList &Args);

/// Parse the -ftapir flag if it is present and get the name of the config file
/// of the Tapir target that was specified. If the argument of the -ftapir flag
/// is invalid, this will return std::nullopt.
std::optional<llvm::StringRef>
getTargetConfigFileName(const llvm::opt::ArgList &Args);

} // namespace clang

#endif
