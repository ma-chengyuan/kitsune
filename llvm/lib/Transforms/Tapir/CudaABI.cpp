//===- CudaABI.cpp - Lower Tapir to the Kitsune CUDA target ----*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===
//
// TODO: add printf() support.
// TODO: double precision device-side entry points.
// TODO: bring high-level design a bit closer to HipABI (as needed).
// TODO: expose enviornment variable target settings???
//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi" // support for -debug-only=cuabi

// NOTES: From the NVPTX target documentation.
//  (See: https://llvm.org/docs/NVPTXUsage.html)
//
//  The NVVMReflect pass should be executed early in the optimization
//  pipeline, immediately after the link stage. The internalize pass is also
//  recommended to remove unused math functions from the resulting PTX. For an
//  input IR module module.bc, the following compilation flow is recommended:
//
//  1. Save list of external functions in module.bc.
//  2. Link module.bc with libdevice.compute_XX.YY.bc.
//  3. Internalize all functions not in list from step #1.
//  4. Eliminate all unused internal functions.
//  5. Run NVVMReflect pass.
//  6. Run standard optimization pipeline.
//

/// ## CudaABI Transformation Command Line Options ##
///
/// All of the transformation's command line options must be
/// passed using `-mllvm` as the leading flag.  All
/// transform options have `-cuabi-` as the leading
/// string.  A summary of the options is provided below.
///
///   * `-cuabi-arch=target`: The target CUDA architecture
///     to generate code for.  This directly matches the
///     [NVPTX backend targets]
///     (https://llvm.org/docs/NVPTXUsage.html).
///
///   * `-cuabi-opt-level=[0,1,2,3]`: Set the optimization
///     level for transformation.  This corresponds directly
///     to standard optimization levels but will be applied
///     to the CUDA device-side code created by the
///     transformation.  Note that this optimization pass
///     occurs *after* an existing (in progress) optimization
///     pipeline has occurred on the original input code
///     module.  This currently defaults to level 3.  This
///     optimization level is also shared with the PTX
///     assembler (ptxas) that comes as part of the CUDA
///     distribution.
///
///   * `-cuabi-host-opt-level=[0,1,2,3]`: Set the optimization
///     level to use for the final host-side module after the CUDA
///     transformation has completed.  Even though the host code
///     has already been through a series of optimizations this
///     option enables a second series of passes over the code
///     after the transformation has completed.  At present there
///     are unlikely to be significant gains from this.  As a
///     result this defaults to level 0, which disables the
///     extra pass entirely.
///
///   * `-cuabi-prefetch`: Enable/Disable the generation of
///     data prefetch calls prior to the kernel launch. This
///     is enabled by default and typically will enable better
///     performance given the current use of managed memory
///     allocations.
///
///   * `-cuabi-max-threads-per-blk`: Set the maximum number
///     of threads that can run within a block.  This limit
///     is coordinated with the runtime's default settings
///     and will place an artificial limit on the thread count.
///     The default behavior is to match the hardware limits
///     within CUDA.
///
///   * `-cuabi-default-grainsize`: EXPERIMENTAL -- control the
///     transform's grain size.  By default this is set to 1 and
///     it is not recommended to change this unless you are
///     extremely familiar with the code generation details and
///     the implications for GPU code execution.
///
///   * `cuabi-keep-files`: The transform has the ability to
///     save the various stages of the IR during execution.
///     In addition, some files are created and removed during
///     execution.  This option will enable all these files to
///     remain (or be created) during execution.  This is
///     obviously helpful if you are debugging the transform.
///
namespace {

const std::string CUABI_PREFIX = "_cuabi";
const std::string CUABI_KERNEL_NAME_PREFIX = CUABI_PREFIX + "_kern_";

// NOTE: At this point in time we do not provide support for the older range
// of GPU architectures. We favor 64-bit and SM_60 or newer, which
// follows the trends of longer term CUDA support.  Although exposed here, we
// have not tested 32-bit host support.
#ifndef _CUDAABI_DEFAULT_TARGET_ARCH
#define _CUDAABI_DEFAULT_TARGET_ARCH "sm_80"
#endif
cl::opt<std::string>
    GPUArch("cuabi-arch", cl::init(_CUDAABI_DEFAULT_TARGET_ARCH), cl::NotHidden,
            cl::desc("Target GPU architecture for CUDA ABI transformation."
                     "(default: " _CUDAABI_DEFAULT_TARGET_ARCH));

cl::opt<unsigned>
    OptLevel("cuabi-opt-level", cl::init(3), cl::NotHidden,
             cl::desc("Specify the GPU kernel optimization level."));

cl::opt<unsigned> HostOptLevel(
    "cuabi-host-opt-level", cl::init(0), cl::NotHidden,
    cl::desc(
        "The optimization level for an experimental pass over the transformed "
        "host-side code."));

cl::opt<bool> CodeGenPrefetch("cuabi-prefetch", cl::init(true), cl::NotHidden,
                              cl::desc("Enable generation of calls to do data "
                                       "prefetching for managed memory."));

cl::opt<bool>
    UseOccupancyLaunches("cuabi-occupancy-launches", cl::init(true),
                         cl::NotHidden,
                         cl::desc("Enable generation of calls to calculate "
                                  "kernel launch parameters based "
                                  "on estimated occupancy calculations."));

const unsigned int CUDAABI_MAX_THREADS_PER_BLOCK = 1024;
const unsigned int CUDAABI_DEFAULT_MAX_THREADS_PER_BLOCK = 256;
cl::opt<unsigned int> MaxThreadsPerBlock(
    "cuabi-max-threads-per-blk",
    cl::init(CUDAABI_DEFAULT_MAX_THREADS_PER_BLOCK), cl::Hidden,
    cl::desc("Set the maximum number of threads per block generated code "
             "can support at execution.\n"));

cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"));

cl::opt<bool> KeepIntermediateFiles(
    "cuabi-keep-files", cl::init(false), cl::Hidden,
    cl::desc("Keep all the intermediate files on disk after"
             "successsful completion of the transforms. (default=false)"));

cl::opt<bool>
    Verbose("cuabi-verbose", cl::init(false), cl::NotHidden,
            cl::desc("Enable verbose mode for cuda toolchain components. "
                     "(default=false)"));

cl::opt<bool> DontFuseReducers(
    "cuabi-dont-fuse-reducers", cl::init(false), cl::NotHidden,
    cl::desc("Generate separate reduction trees for each reducer. "
             "(default=false)"));

cl::opt<bool>
    EmbedPTXInFatbinaries("cuabi-embed-ptx", cl::init(false), cl::Hidden,
                          cl::desc("Embed intermediate PTX files in the "
                                   "fatbinaries used by the CUDA ABI "
                                   "transformation."));

cl::opt<unsigned>
    DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(0), cl::Hidden,
                           cl::desc("Set the runtime system's value for "
                                    "the default number of threads per block. "
                                    "(default: 0 = disabled)"));

cl::opt<unsigned>
    DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), cl::Hidden,
                         cl::desc("Hard-code the runtime system's value for "
                                  "the number of blocks per grid in kernel "
                                  "launches. (default: 0 = disabled)"));

cl::opt<bool> UseKitCudaRuntimeBC(
    "use-kitcuda-runtime-bc", cl::init(true),
    cl::desc("Use a bitcode file for the Kitsune CUDA runtime ABI"),
    cl::Hidden);

cl::opt<std::string> ClKitCudaRuntimeBCPath(
    "kitcuda-runtime-bc-path", cl::init(""),
    cl::desc("Path to the bitcode file for the Kitsune CUDA runtime ABI"),
    cl::Hidden);

// Take the NVIDIA CUDA 'sm_' architecture format and convert it into
// the 'compute_' form.
std::string virtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back some from the full suite of Nvidia targets
  // as we are going in assuming we will support only CUDA 11 or greater.
  // We should probably raise an error for sm_2x and sm_3x targets.
  LLVM_DEBUG(dbgs() << "cuabi: target architecture '" << Arch << "'.\n");
  std::string VirtArch = llvm::StringSwitch<std::string>(Arch)
                             .Case("sm_60", "compute_60") // Pascal
                             .Case("sm_61", "compute_61") //
                             .Case("sm_62", "compute_62") //
                             .Case("sm_70", "compute_70") // Volta
                             .Case("sm_72", "compute_72") //
                             .Case("sm_75", "compute_75") // Turing
                             .Case("sm_80", "compute_80") // Ampere
                             .Case("sm_86", "compute_86") //
                             .Case("sm_87", "compute_87") //
                             .Case("sm_90", "compute_90") // Hopper
                             .Default("unknown");
  LLVM_DEBUG(dbgs() << "cuabi: compute architecture '" << VirtArch << "'.\n");
  if (VirtArch == "unknown") {
    errs() << "cuabi: unsupported cuda architecture target '" << Arch << "'.\n";
    report_fatal_error("cuabi: fatal error -- unsupported target.");
  }
  return VirtArch;
}

std::string PTXVersionFromCudaVersion() {
  std::string CudaVersion;
  raw_string_ostream ss(CudaVersion);
  
  // Convert macros to strings to check if they're empty
  #define STRINGIFY(x) #x
  #define TO_STRING(x) STRINGIFY(x)
  
  std::string major_str = TO_STRING(CUDATOOLKIT_VERSION_MAJOR);
  std::string minor_str = TO_STRING(CUDATOOLKIT_VERSION_MINOR);
  
  // Remove quotes if present (when macro is empty string "")
  if (major_str.size() >= 2 && major_str[0] == '"' && major_str.back() == '"') {
    major_str = major_str.substr(1, major_str.size() - 2);
  }
  if (minor_str.size() >= 2 && minor_str[0] == '"' && minor_str.back() == '"') {
    minor_str = minor_str.substr(1, minor_str.size() - 2);
  }
  
  // Check if we have valid version numbers
  if (!major_str.empty() && !minor_str.empty()) {
    ss << major_str << "." << minor_str;
  } else {
    // Default to CUDA 12.0 if version not provided
    ss << "12.0";
  }
  
  #undef TO_STRING
  #undef STRINGIFY
  LLVM_DEBUG(dbgs() << "cuabi: cuda toolkit version: " << CudaVersion << "\n");

  std::string PTXVersionStr =
      llvm::StringSwitch<std::string>(CudaVersion)
          // TODO: These CUDA to PTX version translations will have
          // to be watched between CUDA and LLVM resources.  It is
          // not uncommon for LLVM to lag well behind CUDA PTX versions.
          // The details below are based on Cuda 12.3 and LLVM 16.
          .Case("10.0", "+ptx63")
          .Case("10.1", "+ptx64")
          .Case("10.2", "+ptx65")
          .Case("10.0", "+ptx63")
          .Case("11.0", "+ptx70")
          .Case("11.1", "+ptx71")
          .Case("11.2", "+ptx72")
          .Case("11.3", "+ptx72")
          .Case("11.4", "+ptx72")
          .Case("11.5", "+ptx72")
          .Case("11.6", "+ptx76")
          .Case("11.7", "+ptx77")
          .Case("11.8", "+ptx78")
          .Case("12.0", "+ptx83")
          .Case("12.1", "+ptx83")
          .Case("12.2", "+ptx83")
          .Case("12.3", "+ptx83")
          .Case("12.4", "+ptx83")
          .Case("12.5", "+ptx83")
          .Case("12.8", "+ptx83")
          .Case("12.9", "+ptx83")
          .Case("13.0", "+ptx83")
          .Case("13.1", "+ptx83")
          .Default("");

  if (PTXVersionStr == "") {
    errs() << "cuabi: cuda toolkit version: " << CudaVersion << "\n";
    report_fatal_error("cuabi: cuda --> ptx version mapping is out-of-date.");
  }

  LLVM_DEBUG(dbgs() << "cuabi: target ptx version: " << PTXVersionStr << "\n");
  return PTXVersionStr;
}

// Copied from NVPTXAssignValidGlobalNames::cleanUpName
std::string cleanUpName(StringRef Name) {
  std::string ValidName;
  raw_string_ostream ValidNameStream(ValidName);
  for (char C : Name) {
    // While PTX also allows '%' at the start of identifiers, LLVM will throw a
    // fatal error for '%' in symbol names in MCSymbol::print. Exclude for now.
    if (isAlnum(C) || C == '_' || C == '$') {
      ValidNameStream << C;
    } else {
      ValidNameStream << "_$_";
    }
  }
  return ValidNameStream.str();
}

std::pair<DebugLoc, DebugLoc> getFirstAndLastDebugLoc(Function *F) {
  DebugLoc FirstLoc;
  DebugLoc LastLoc;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto Loc = I.getDebugLoc()) {
        unsigned Line = Loc->getLine();
        unsigned Col = Loc->getColumn();
        if (!FirstLoc || Line < FirstLoc->getLine() ||
            (Line == FirstLoc->getLine() && Col < FirstLoc->getColumn())) {
          FirstLoc = Loc;
        }
        if (!LastLoc || Line > LastLoc->getLine() ||
            (Line == LastLoc->getLine() && Col > LastLoc->getColumn())) {
          LastLoc = Loc;
        }
      }
    }
  }
  return {FirstLoc, LastLoc};
}

} // namespace

/// Check GPU parallelism level from loop metadata.
/// Returns true if this loop has threadparallel pragma (inner loop).
/// Returns false if this loop has blockparallel pragma (outer loop) or no pragma.
static bool isThreadParallelLoop(const Loop *L) {
  MDNode *LoopID = L->getLoopID();
  if (!LoopID)
    return false;
  
  // Check for tapir.loop.gpu.threadparallel metadata
  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;
    
    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;
    
    if (S->getString() == "tapir.loop.gpu.threadparallel") {
      LLVM_DEBUG(dbgs() << "cuabi: detected threadparallel loop\n");
      return true;
    }
  }
  return false;
}

// Helper function to configure the details of our post-Tapir transformation
// passes.

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.  TODO: This is
/// a not so great naming mechanism and certainly not thread safe...
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, Module &KernelModule, const std::string &KN,
                   CudaABI *T, bool MakeUniqueName)
    : LoopOutlineProcessor(M, KernelModule), TTarget(T), KernelName(KN),
      KernelModule(KernelModule) {

  if (MakeUniqueName) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID++;
    KernelName = UN;
  }

  LLVM_DEBUG(dbgs() << "cuabi: creating loop outliner:\n"
                    << "\tkernel name: " << KernelName << "\n"
                    << "\tmodule     : " << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);

  // NVVM-centric barrier -- equivalent to Cuda's __sync_threads().
  CUSyncThreads =
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::nvvm_barrier0);

  // Get entry points into the Cuda-centric portion of the Kitsune GPU runtime.
  KernelInstMixTy = StructType::get(Int64Ty,  // number of memory ops.
                                    Int64Ty,  // number of floating point ops.
                                    Int64Ty); // number of integer ops.
  KitCudaLaunchFn = M.getOrInsertFunction(
      "__kitcuda_launch_kernel",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrPtrTy,                    // arguments
      Int64Ty,                         // trip count
      Int32Ty,                         // threads-per-block
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque cuda stream

  KitCudaLaunchNestedFn = M.getOrInsertFunction(
      "__kitcuda_launch_kernel_nested",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrPtrTy,                    // arguments
      Int64Ty,                         // num_blocks (outer loop trips)
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque cuda stream

  KitCudaMemPrefetchFn =
      M.getOrInsertFunction("__kitcuda_mem_gpu_prefetch",
                            VoidPtrTy,  // return an opaque stream
                            VoidPtrTy,  // pointer to prefetch
                            VoidPtrTy); // opaque stream
  KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // return the device pointer for symbol.
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name
  KitCudaMemcpySymbolToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_symbol_to_device",
                            VoidTy,   // no return
                            Int32Ty,  // host pointer
                            Int64Ty,  // device pointer
                            Int64Ty); // number of bytes to copy

  KitCudaMemAllocAndCopyToDevFn = M.getOrInsertFunction(
      "__kitcuda_mem_alloc_and_copy_to_device", VoidPtrTy, // return the stream
      VoidPtrTy,                                           // host pointer
      Int64Ty,      // number of bytes to copy
      VoidPtrPtrTy, // device pointer
      VoidPtrTy);   // opaque stream

  LLVM_DEBUG(dbgs() << "\t\tdone.\n");
}

CudaLoop::~CudaLoop() {}

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {

  // Add the loop control inputs -- the first parameter defines
  // the extent of the index space (the number of threads to launch).
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the
  // index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grain size, if it is
  // not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("grainSize");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays
  // used in the loop body).
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
  }
}

unsigned CudaLoop::getIVArgIndex(const Function &F,
                                 const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned CudaLoop::getLimitArgIndex(const Function &F,
                                    const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

static std::set<GlobalValue *> &collect(Constant &c,
                                        std::set<GlobalValue *> &seen);

static std::set<GlobalValue *> &collect(BasicBlock &bb,
                                        std::set<GlobalValue *> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(Function &f,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&f);

  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalVariable &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalIFunc &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalAlias &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(BlockAddress &blkaddr,
                                        std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collect(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collect(*bb, seen);
  return seen;
}

std::set<GlobalValue *> &collect(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (auto *f = dyn_cast<Function>(&c))
    return collect(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collect(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collect(*blkaddr, seen);
  else
    for (auto &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collect(*cop, seen);
  return seen;
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  // TODO: process loop prior to outlining to do GPU/CUDA-specific things
  // like capturing global variables, etc.
  LLVM_DEBUG(dbgs() << "\tcuabi: preprocessing loop for kernel '" << KernelName
                    << "', in module '" << KernelModule.getName() << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it will
  // need to be cloned into the KernelModule and then register with CUDA
  // in the CUDA-centric ctor.
  LLVM_DEBUG(dbgs() << "\t\t- gathering and analyzing global values...\n");
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();

  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  // Clone global variables (TODO: and aliases).
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // TODO: Make sure this logic makes sense...
      // We don't necessarily need a GPU-side clone of a
      // global variable -- instead we need a location where
      // we can copy symbol information from the host.
      GlobalVariable *NewGV = nullptr;
      // If GV is a constant we can clone the entire
      // variable over, including the initializer
      // details, and deal with it as an internal
      // variable (i.e., no need to coordinate with
      // host).  TODO: make sure this is sound!
      if (GV->isConstant())
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ true,
            GlobalValue::InternalLinkage, GV->getInitializer(),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);
      else {
        // If GV is non-constant we will need to
        // create a device-side version that will
        // have the host-side value copied over
        // prior to launching the cooresponding
        // kernel...
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ false,
            GlobalValue::ExternalWeakLinkage,
            (Constant *)Constant::getNullValue(GV->getValueType()),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);
        TTarget->pushGlobalVariable(GV);
      }
      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
      LLVM_DEBUG(dbgs() << "\t\t\tcreated kernel-side global variable '"
                        << NewGV->getName() << "'.\n");
    } else if (dyn_cast<GlobalAlias>(V))
      llvm_unreachable("cuabi: fatal error, GlobalAlias not implemented!");
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        // LLVM_DEBUG(dbgs() << "\tanalyzing missing (device-side) function '"
        //                   << F->getName() << "'.\n");
        Function *LF = resolveLibDeviceFunction(F, false);
        if (LF && not KernelModule.getFunction(LF->getName())) {
          // LLVM_DEBUG(dbgs() << "\ttransformed to libdevice function '"
          //                   << LF->getName() << "'.\n");
          DeviceF = Function::Create(LF->getFunctionType(), F->getLinkage(),
                                     LF->getName(), KernelModule);
        } else {
          // LLVM_DEBUG(dbgs() << "\tcreated device function '" << F->getName()
          //                   << "'.\n");
          DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                     F->getName(), KernelModule);
        }
      }

      for (size_t i = 0; i < F->arg_size(); i++) {
        Argument *Arg = F->getArg(i);
        Argument *NewA = DeviceF->getArg(i);
        NewA->setName(Arg->getName());
        VMap[Arg] = NewA;
      }
      VMap[F] = DeviceF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues) {
    if (dyn_cast<GlobalIFunc>(V)) {
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size() && not F->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        // Only clone if function does not yet have a body
        if (DeviceF->size() == 0) {
          CloneFunctionInto(DeviceF, F, VMap,
                            CloneFunctionChangeType::DifferentModule, Returns);
          LLVM_DEBUG(dbgs() << "cuabi: cloning device function '"
                            << DeviceF->getName() << "' into kernel module.\n");
        }
        // GPU calls are slow, try to force inlining...
        if (OptLevel > 1 && not DeviceF->hasFnAttribute(Attribute::NoInline))
          DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\tfinished preprocessing tapir loop.\n");
  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> PreLoopIRFile;
    SmallString<255> IRFileName("preprocess-loop.ll");
    PreLoopIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    M.print(PreLoopIRFile->os(), nullptr);
    PreLoopIRFile->keep();
  }
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  // addSyncToOutlineReturns(TLI, Out, VMap);

  LLVMContext &Ctx = M.getContext();
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  TapirLoopHints Hints(TL);

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);

  TTarget->pushSR(T->getDetach()->getSyncRegion());

  // We no longer need the cloned sync region.
  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up
  // any stray (target related) attributes that were
  // attached as part of the host-side target that
  // occurred before outlining.
  Function *KernelF = Out.Outline;
  KernelF->setName(KernelName);
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("personality");
  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("target-features",
                     PTXVersionFromCudaVersion() + "," + GPUArch);
  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  // AV.push_back(MDString::get(Ctx, "maxntidx"));
  // AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
  // 160)));
  // AV.push_back(MDString::get(Ctx, "maxnreg"));
  // AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
  // 63)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Mark pointer arguments as readonly/noalias when they are only used for
  // loads (via GEP chains). This enables the NVPTX backend to use
  // ld.global.nc (read-only / texture cache) for read-only data like twiddle
  // factor tables.
  for (unsigned I = 0, E = KernelF->arg_size(); I < E; ++I) {
    Argument *Arg = KernelF->getArg(I);
    if (!Arg->getType()->isPointerTy())
      continue;
    bool OnlyLoads = true;
    bool HasLoad = false;
    SmallVector<Value *, 16> Worklist;
    SmallPtrSet<Value *, 16> Visited;
    Worklist.push_back(Arg);
    while (!Worklist.empty() && OnlyLoads) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;
      for (User *UU : V->users()) {
        if (auto *LI = dyn_cast<LoadInst>(UU)) {
          HasLoad = true;
        } else if (auto *SI = dyn_cast<StoreInst>(UU)) {
          if (SI->getPointerOperand() == V) {
            OnlyLoads = false;
            break;
          }
        } else if (isa<GetElementPtrInst>(UU)) {
          Worklist.push_back(UU);
        } else {
          // Unknown use (call, atomicrmw, etc.) — be conservative
          OnlyLoads = false;
          break;
        }
      }
    }
    if (OnlyLoads && HasLoad) {
      KernelF->addParamAttr(I, Attribute::ReadOnly);
      KernelF->addParamAttr(I, Attribute::NoAlias);
      LLVM_DEBUG(dbgs() << "cuabi: marked kernel arg #" << I
                        << " as readonly noalias\n");
    }
  }

  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We really only want a grainsize of 1 for now...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC
    // arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> B(Entry->getTerminator());

  // Check if this kernel contains nested tapir loops by looking for detach
  // instructions. If present, the outer loop uses blockIdx and inner loops
  // use threadIdx.
  bool HasNestedThreadLoops = false;
  int DetachCount = 0;

  for (BasicBlock &BB : *KernelF)
    for (Instruction &I : BB)
      if (isa<DetachInst>(&I))
        DetachCount++;

  if (DetachCount > 0) {
    HasNestedThreadLoops = true;
    MDNode *NestedMD = MDNode::get(Ctx, MDString::get(Ctx, "cuda.kernel.nested"));
    KernelF->setMetadata("cuda.kernel.nested", NestedMD);
    LLVM_DEBUG(dbgs() << "cuabi: nested parallelism detected (" << DetachCount
                      << " detach(es)), using block/thread split\n");
  }

  Value *BlockIV;
  if (HasNestedThreadLoops) {
    // This is a block-parallel loop - use only blockIdx for the outer loop IV
    LLVM_DEBUG(dbgs() << "cuabi: generating block-level parallelism (blockIdx only)\n");
    Value *BlockIdx = B.CreateCall(CUBlockIdxX);
    BlockIV = B.CreateIntCast(BlockIdx, PrimaryIV->getType(), false, "block_iv");
    
    // Block loop: each block handles one iteration
    Value *BlockEnd = B.CreateAdd(BlockIV, Grainsize, "block_end");
    Value *Cond = B.CreateICmpUGE(BlockIV, End, "cond_block_end");
    ReplaceInstWithInst(Entry->getTerminator(),
                        BranchInst::Create(Exit, Header, Cond));
    
    // Use the block ID as the start iteration number for the primary IV.
    // IMPORTANT: For nested loops, we must ONLY update the PHI node, not all uses!
    // Using replaceAllUsesWith would break inner loops that might use the same value.
    PrimaryIV->setIncomingValueForBlock(Entry, BlockIV);
    
    // Update cloned loop condition to use the block-end value.
    unsigned TripCountIdx = 0;
    ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
    if (ClonedCond->getOperand(0) != End)
      ++TripCountIdx;
    assert(ClonedCond->getOperand(TripCountIdx) == End &&
           "End argument not used in condition!");
    ClonedCond->setOperand(TripCountIdx, BlockEnd);
  } else {
    // Standard single-level parallel loop - use full thread ID calculation
    LLVM_DEBUG(dbgs() << "cuabi: generating thread-level parallelism (blockIdx + threadIdx)\n");
    
    // This is the classic CUDA thread ID calculation:
    //      i = blockDim.x * blockIdx.x + threadIdx.x;
    // For now we only generate 1-D thread IDs.
    Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
    Value *BlockIdx = B.CreateCall(CUBlockIdxX);
    Value *BlockDim = B.CreateCall(CUBlockDimX);
    Value *ThreadIV = B.CreateIntCast(
        B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim, "blk_offset"),
                    "cuthread_id"),
        PrimaryIV->getType(), false, "thread_iv");

    // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the
    // current codegen...
    // ThreadID = B.CreateMul(ThreadID, Grainsize);
    Value *ThreadEnd = B.CreateAdd(ThreadIV, Grainsize, "thread_end");
    Value *Cond = B.CreateICmpUGE(ThreadIV, End, "cond_thread_end");
    ReplaceInstWithInst(Entry->getTerminator(),
                        BranchInst::Create(Exit, Header, Cond));

    // Use the thread ID as the start iteration number for the primary IV.
    PrimaryIVInput->replaceAllUsesWith(ThreadIV);
    
    // Update cloned loop condition to use the thread-end value.
    unsigned TripCountIdx = 0;
    ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
    if (ClonedCond->getOperand(0) != End)
      ++TripCountIdx;
    assert(ClonedCond->getOperand(TripCountIdx) == End &&
           "End argument not used in condition!");
    ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  }

  // For reducers and scanners, we need ThreadIdx and BlockDim values
  // Ensure they're available regardless of loop type
  // IMPORTANT: Create a new IRBuilder because we replaced Entry's terminator above
  IRBuilder<> B2(Entry->getTerminator());
  Value *ThreadIdx = B2.CreateCall(CUThreadIdxX);
  Value *BlockDim = B2.CreateCall(CUBlockDimX);

  fixReducersInKernel(KernelF, ThreadIdx, BlockDim, VMap);
  fixScannersInKernel(KernelF, End, VMap);
  
  // If this kernel has nested thread-parallel loops, transform them now
  if (HasNestedThreadLoops) {
    transformThreadParallelLoops(KernelF, BlockIV, VMap);
  }
  
  // Strip llvm.loop.unroll.disable metadata from ALL loops in the kernel,
  // even for non-nested kernels. This prevents the NVPTX backend from emitting
  // .pragma "nounroll" which blocks ptxas from optimizing GPU loops.
  for (BasicBlock &BB : *KernelF) {
    Instruction *Term = BB.getTerminator();
    if (MDNode *LoopMD = Term->getMetadata(LLVMContext::MD_loop)) {
      bool HasUnrollDisable = false;
      for (unsigned i = 1, e = LoopMD->getNumOperands(); i < e; ++i) {
        if (MDNode *Op = dyn_cast<MDNode>(LoopMD->getOperand(i))) {
          if (Op->getNumOperands() > 0) {
            if (MDString *S = dyn_cast<MDString>(Op->getOperand(0))) {
              if (S->getString() == "llvm.loop.unroll.disable") {
                HasUnrollDisable = true;
                break;
              }
            }
          }
        }
      }
      if (HasUnrollDisable) {
        Term->setMetadata(LLVMContext::MD_loop, nullptr);
      }
    }
  }
  
  fixDebugInfoInKernel(KernelF);

  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> PostLoopIRFile;
    SmallString<255> IRFileName("post-loop.ll");
    PostLoopIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    KernelModule.print(PostLoopIRFile->os(), nullptr);
    PostLoopIRFile->keep();
  }
}

void CudaLoop::transformThreadParallelLoops(Function *KernelF, Value *BlockIV,
                                            ValueToValueMapTy &VMap) {
  LLVM_DEBUG(dbgs() << "cuabi: transforming nested thread-parallel loops in kernel '"
                    << KernelF->getName() << "'\n");

  // Find all leaf tapir loops (have detach but no nested tapir sub-loops).
  SmallVector<Loop *, 4> ThreadParallelLoops;
  DominatorTree DT(*KernelF);
  LoopInfo LI(DT);

  std::function<void(Loop*)> ProcessLoop = [&](Loop *L) {
    bool HasDetach = false;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (isa<DetachInst>(&I)) { HasDetach = true; break; }
      }
      if (HasDetach) break;
    }

    bool HasNestedTapirLoop = false;
    for (Loop *SubL : L->getSubLoops()) {
      for (BasicBlock *BB : SubL->blocks()) {
        for (Instruction &I : *BB) {
          if (isa<DetachInst>(&I)) { HasNestedTapirLoop = true; break; }
        }
        if (HasNestedTapirLoop) break;
      }
      if (HasNestedTapirLoop) break;
    }

    if (HasDetach && !HasNestedTapirLoop) {
      ThreadParallelLoops.push_back(L);
      LLVM_DEBUG(dbgs() << "  found leaf tapir loop at header "
                        << L->getHeader()->getName() << "\n");
    }

    for (Loop *SubL : L->getSubLoops())
      ProcessLoop(SubL);
  };

  for (Loop *L : LI)
    ProcessLoop(L);

  LLVM_DEBUG(dbgs() << "  " << ThreadParallelLoops.size()
                    << " thread-parallel loop(s) to transform\n");

  // Transform each thread-parallel loop.
  for (Loop *L : ThreadParallelLoops) {
    BasicBlock *Header = L->getHeader();
    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Latch = L->getLoopLatch();

    if (!Header || !Latch)
      continue;

    if (!Preheader) {
      Preheader = InsertPreheaderForLoop(L, &DT, &LI, nullptr, true);
      if (!Preheader)
        continue;
    }

    // Find the induction variable PHI node.
    PHINode *IV = nullptr;
    for (PHINode &Phi : Header->phis()) {
      if (Phi.getType()->isIntegerTy() && Phi.getBasicBlockIndex(Preheader) >= 0) {
        IV = &Phi;
        break;
      }
    }

    if (!IV) {
      LLVM_DEBUG(dbgs() << "  warning: no suitable IV PHI found, skipping loop\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "  transforming loop with IV: " << *IV << "\n");

    Value *Start = IV->getIncomingValueForBlock(Preheader);
    Value *Next = IV->getIncomingValueForBlock(Latch);

    // If the start value was incorrectly set to block_iv, reset to 0.
    if (Start->hasName() && Start->getName().starts_with("block_iv")) {
      Start = ConstantInt::get(IV->getType(), 0);
      IV->setIncomingValueForBlock(Preheader, Start);
    }

    // Each thread starts at: start + threadIdx.x, strides by blockDim.x.
    IRBuilder<> B(Preheader->getTerminator());
    Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
    Value *ThreadIV = B.CreateIntCast(ThreadIdx, IV->getType(), false, "thread_idx");
    Value *ThreadIterator = B.CreateAdd(Start, ThreadIV, "thread_iter");
    IV->setIncomingValueForBlock(Preheader, ThreadIterator);

    // Change stride from 1 to blockDim.x.
    if (Instruction *IncInst = dyn_cast<Instruction>(Next)) {
      if (IncInst->getOpcode() == Instruction::Add) {
        Value *OtherOp = nullptr;
        if (IncInst->getOperand(0) == IV)
          OtherOp = IncInst->getOperand(1);
        else if (IncInst->getOperand(1) == IV)
          OtherOp = IncInst->getOperand(0);

        if (OtherOp) {
          IRBuilder<> LatchBuilder(IncInst);
          Value *BlockDim = LatchBuilder.CreateCall(CUBlockDimX);
          Value *BlockDimCast = LatchBuilder.CreateIntCast(BlockDim, IV->getType(), false, "blockDim_x");
          if (IncInst->getOperand(0) == IV)
            IncInst->setOperand(1, BlockDimCast);
          else
            IncInst->setOperand(0, BlockDimCast);
          LLVM_DEBUG(dbgs() << "  stride changed to blockDim.x\n");
        }
      }
    }

    // Fix exit condition: change == to >= so strided loop terminates correctly.
    BasicBlock *ExitingBB = L->getExitingBlock();
    if (!ExitingBB)
      ExitingBB = Latch;
    if (ExitingBB) {
      BranchInst *ExitBranch = dyn_cast<BranchInst>(ExitingBB->getTerminator());
      if (ExitBranch && ExitBranch->isConditional()) {
        ICmpInst *ExitCond = dyn_cast<ICmpInst>(ExitBranch->getCondition());
        if (ExitCond) {
          ICmpInst::Predicate OldPred = ExitCond->getPredicate();
          ICmpInst::Predicate NewPred = OldPred;
          bool ExitOnTrue = !L->contains(ExitBranch->getSuccessor(0));

          if (ExitOnTrue) {
            if (OldPred == ICmpInst::ICMP_EQ)  NewPred = ICmpInst::ICMP_SGE;
            else if (OldPred == ICmpInst::ICMP_UGT) NewPred = ICmpInst::ICMP_UGE;
            else if (OldPred == ICmpInst::ICMP_SGT) NewPred = ICmpInst::ICMP_SGE;
          } else {
            if (OldPred == ICmpInst::ICMP_NE)  NewPred = ICmpInst::ICMP_SLT;
          }

          if (NewPred != OldPred)
            ExitCond->setPredicate(NewPred);
        }
      }
    }

    // Add a preheader guard to skip threads whose start index is past the bound.
    {
      Value *LoopBound = nullptr;
      if (ExitingBB) {
        BranchInst *ExitBranch = dyn_cast<BranchInst>(ExitingBB->getTerminator());
        if (ExitBranch && ExitBranch->isConditional()) {
          ICmpInst *ExitCond = dyn_cast<ICmpInst>(ExitBranch->getCondition());
          if (ExitCond) {
            if (ExitCond->getOperand(0) == Next)
              LoopBound = ExitCond->getOperand(1);
            else if (ExitCond->getOperand(1) == Next)
              LoopBound = ExitCond->getOperand(0);
          }
        }
      }

      if (LoopBound) {
        IRBuilder<> GuardBuilder(Preheader->getTerminator());
        Value *BoundCast = LoopBound;
        if (LoopBound->getType() != ThreadIterator->getType())
          BoundCast = GuardBuilder.CreateIntCast(LoopBound, ThreadIterator->getType(), true, "bound_cast");

        Value *GuardCond = GuardBuilder.CreateICmpSGE(ThreadIterator, BoundCast, "thread_guard");

        SmallVector<BasicBlock *, 4> ExitBlocksForGuard;
        L->getExitBlocks(ExitBlocksForGuard);

        if (!ExitBlocksForGuard.empty()) {
          BasicBlock *ExitBB = ExitBlocksForGuard[0];
          Instruction *OldTerm = Preheader->getTerminator();
          BranchInst::Create(ExitBB, Header, GuardCond, OldTerm);
          OldTerm->eraseFromParent();
        }
      }
    }

    // Serialize Tapir instructions (detach/reattach/sync) in the loop body.
    SmallVector<Instruction *, 4> TapirInstsToRemove;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (isa<DetachInst>(&I) || isa<ReattachInst>(&I) || isa<SyncInst>(&I))
          TapirInstsToRemove.push_back(&I);
      }
    }

    for (Instruction *I : TapirInstsToRemove) {
      if (auto *DI = dyn_cast<DetachInst>(I)) {
        BranchInst::Create(DI->getDetached(), DI);
        DI->eraseFromParent();
      } else if (auto *RI = dyn_cast<ReattachInst>(I)) {
        BranchInst::Create(RI->getSuccessor(0), RI);
        RI->eraseFromParent();
      } else if (auto *SI = dyn_cast<SyncInst>(I)) {
        BranchInst::Create(SI->getSuccessor(0), SI);
        SI->eraseFromParent();
      }
    }

    // Insert __syncthreads() at each loop exit to synchronize threads.
    SmallVector<BasicBlock *, 4> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (BasicBlock *ExitBB : ExitBlocks) {
      IRBuilder<> SyncBuilder(&*ExitBB->getFirstInsertionPt());
      SyncBuilder.CreateCall(CUSyncThreads);
      LLVM_DEBUG(dbgs() << "  inserted __syncthreads() in: " << ExitBB->getName() << "\n");
    }

    // Store the inner loop's constant trip count as kernel metadata for launch config.
    if (ExitingBB) {
      BranchInst *ExitBr = dyn_cast<BranchInst>(ExitingBB->getTerminator());
      if (ExitBr && ExitBr->isConditional()) {
        if (ICmpInst *ExitCmp = dyn_cast<ICmpInst>(ExitBr->getCondition())) {
          Value *Bound = nullptr;
          if (ExitCmp->getOperand(0) == Next)
            Bound = ExitCmp->getOperand(1);
          else if (ExitCmp->getOperand(1) == Next)
            Bound = ExitCmp->getOperand(0);
          if (Bound) {
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Bound)) {
              uint64_t InnerBound = CI->getZExtValue();
              LLVMContext &Ctx = KernelF->getContext();
              MDNode *BoundMD = MDNode::get(Ctx, ConstantAsMetadata::get(
                  ConstantInt::get(Type::getInt64Ty(Ctx), InnerBound)));
              KernelF->setMetadata("cuda.kernel.inner_bound", BoundMD);
            }
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "  thread-parallel loop transformed\n");
  }

  // Strip llvm.loop.unroll.disable metadata from all loops so ptxas can
  // optimize tight inner loops.
  for (BasicBlock &BB : *KernelF) {
    Instruction *Term = BB.getTerminator();
    if (MDNode *LoopMD = Term->getMetadata(LLVMContext::MD_loop)) {
      for (unsigned i = 1, e = LoopMD->getNumOperands(); i < e; ++i) {
        if (MDNode *Op = dyn_cast<MDNode>(LoopMD->getOperand(i))) {
          if (Op->getNumOperands() > 0) {
            if (MDString *S = dyn_cast<MDString>(Op->getOperand(0))) {
              if (S->getString() == "llvm.loop.unroll.disable") {
                Term->setMetadata(LLVMContext::MD_loop, nullptr);
                break;
              }
            }
          }
        }
      }
    }
  }
  
  LLVM_DEBUG({
    dbgs() << "\n=== Kernel IR after thread transformation ===\n";
    KernelF->print(dbgs());
    dbgs() << "\n=== End kernel IR ===\n\n";
  });
}

void CudaLoop::fixDebugInfoInKernel(Function *KernelF) {
  DISubprogram *OldSP = KernelF->getSubprogram();
  if (!OldSP) {
    return;
  }
  // Referenced from:
  // https://github.com/llvm/llvm-project/commit/c10d0e5ccd12f049bddb24dcf8bbb7fbbc6c68f2
  DISubprogram *NewSP = DISubprogram::getDistinct(
      KernelF->getContext(), OldSP->getScope(), OldSP->getName(),
      KernelF->getName(), OldSP->getFile(), OldSP->getLine(), OldSP->getType(),
      OldSP->getScopeLine(), OldSP->getContainingType(),
      OldSP->getVirtualIndex(), OldSP->getThisAdjustment(), OldSP->getFlags(),
      OldSP->getSPFlags(), OldSP->getUnit(), OldSP->getTemplateParams(),
      OldSP->getDeclaration(), OldSP->getRetainedNodes(),
      OldSP->getThrownTypes(), OldSP->getAnnotations(),
      OldSP->getTargetFuncName());
  DenseMap<const MDNode *, MDNode *> Cache;
  const auto ReparentScope = [&](LLVMContext &Ctx, DIScope *Scope) {
    SmallVector<DIScope *, 3> ScopeChain;
    DIScope *Last = NewSP;
    DIScope *CurScope = Scope;
    do {
      if (auto *SP = dyn_cast<DISubprogram>(CurScope)) {
        // Don't rewrite this scope chain if it doesn't lead to the replaced SP.
        if (SP != OldSP)
          return Scope;
        Cache.insert({OldSP, NewSP});
        break;
      }
      if (auto *Found = Cache[CurScope]) {
        Last = cast<DIScope>(Found);
        break;
      }
      ScopeChain.push_back(CurScope);
    } while ((CurScope = CurScope->getScope()));
    // Starting from the top, rebuild the nodes to point to the new inlined-at
    // location (then rebuilding the rest of the chain behind it) and update the
    // map of already-constructed inlined-at nodes.
    for (const DIScope *MD : reverse(ScopeChain)) {
      if (auto *LB = dyn_cast<DILexicalBlock>(MD))
        Cache[MD] = Last = DILexicalBlock::getDistinct(
            Ctx, Last, LB->getFile(), LB->getLine(), LB->getColumn());
      else if (auto *LB = dyn_cast<DILexicalBlockFile>(MD))
        Cache[MD] = Last = DILexicalBlockFile::getDistinct(
            Ctx, Last, LB->getFile(), LB->getDiscriminator());
      else
        llvm_unreachable("illegal parent scope");
    }
    return Last;
  };
  const auto ReparentLocation = [&](LLVMContext &Ctx, DILocation *DL) {
    DILocation *InlinedAt = DL->getInlinedAt();
    if (InlinedAt) {
      while (auto *IA = InlinedAt->getInlinedAt())
        InlinedAt = IA;
      DIScope *NewScope = ReparentScope(Ctx, InlinedAt->getScope());
      InlinedAt = DILocation::get(Ctx, InlinedAt->getLine(),
                                  InlinedAt->getColumn(), NewScope);
    }
    return DILocation::get(
        Ctx, DL->getLine(), DL->getColumn(), ReparentScope(Ctx, DL->getScope()),
        DebugLoc::appendInlinedAt(DL, InlinedAt, Ctx, Cache));
  };
  const auto ReparentDebugInfo = [&](Instruction &I) {
    auto DL = I.getDebugLoc();
    auto &Ctx = I.getContext();
    if (DL)
      I.setDebugLoc(ReparentLocation(Ctx, DL.get()));
    if (auto *MD = I.getMetadata(LLVMContext::MD_loop)) {
      for (unsigned I = 1; I < MD->getNumOperands(); I++) {
        if (auto *DIL = dyn_cast_or_null<DILocation>(MD->getOperand(I))) {
          MD->replaceOperandWith(I, ReparentLocation(Ctx, DIL));
        }
      }
    }
    // Fix up debug variables to point to NewSP.
    const auto ReparentVar = [&](DILocalVariable *Var) {
      return DILocalVariable::getDistinct(
          Ctx, cast<DILocalScope>(ReparentScope(Ctx, Var->getScope())),
          Var->getName(), Var->getFile(), Var->getLine(), Var->getType(),
          Var->getArg(), Var->getFlags(), Var->getAlignInBits(),
          Var->getAnnotations());
    };
    if (auto *DbgValue = dyn_cast<DbgValueInst>(&I)) {
      auto *Var = DbgValue->getVariable();
      I.setOperand(2, MetadataAsValue::get(Ctx, ReparentVar(Var)));
    } else if (auto *DbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
      auto *Var = DbgDeclare->getVariable();
      I.setOperand(1, MetadataAsValue::get(Ctx, ReparentVar(Var)));
    }
  };

  KernelF->setSubprogram(NewSP);
  for (BasicBlock &BB : *KernelF) {
    for (Instruction &I : BB) {
      ReparentDebugInfo(I);
    }
  }
}

namespace {
const size_t WarpSize = 32;
const size_t MaxNumWarps = 32;

BasicBlock *getUniqueExitBlock(Function *KernelF) {
  SmallSet<BasicBlock *, 8> Terminators;
  for (BasicBlock &BB : *KernelF) {
    if (isa<ReturnInst>(BB.getTerminator())) {
      Terminators.insert(&BB);
    }
  }
  if (Terminators.size() == 1) {
    return *Terminators.begin();
  }
  // This branch is probably here just for correctness. It seems that Kernel
  // functions outlined by LoopSpawning already have a single exit block. But
  // in the rare event that it does not, create a new exit block and link all
  // terminators to it.
  BasicBlock *NewExit =
      BasicBlock::Create(KernelF->getContext(), "exit", KernelF);
  for (BasicBlock *T : Terminators) {
    ReplaceInstWithInst(T->getTerminator(), BranchInst::Create(NewExit));
  }
  return NewExit;
}

void createPrintf(IRBuilder<> &B, const char *Fmt, ArrayRef<Value *> Args) {
  Module &M = *B.GetInsertBlock()->getModule();
  // Create a printf on CUDA
  Function *Printf;
  if (auto *F = M.getFunction("vprintf")) {
    Printf = F;
  } else {
    FunctionType *PrintfTy =
        FunctionType::get(B.getInt32Ty(),
                          {PointerType::getUnqual(M.getContext()),
                           PointerType::getUnqual(M.getContext())},
                          false);
    Printf =
        Function::Create(PrintfTy, GlobalValue::ExternalLinkage, "vprintf", &M);
  }
  size_t TotalSize = 0;
  size_t Alignment = 1;
  for (Value *Arg : Args) {
    size_t ArgSize = Arg->getType()->getPrimitiveSizeInBits() / 8;
    TotalSize += ArgSize;
    Alignment = std::max(Alignment, ArgSize);
  }
  Type *ArgType = ArrayType::get(Type::getInt8Ty(M.getContext()), TotalSize);
  AllocaInst *ArgPtr = B.CreateAlloca(ArgType);
  ArgPtr->setAlignment(Align(Alignment));
  size_t Offset = 0;
  for (Value *Arg : Args) {
    size_t ArgSize = Arg->getType()->getPrimitiveSizeInBits() / 8;
    B.CreateStore(Arg,
                  B.CreateConstInBoundsGEP2_32(ArgType, ArgPtr, 0, Offset));
    Offset += ArgSize;
  }
  B.CreateCall(Printf, {B.CreateGlobalStringPtr(Fmt), ArgPtr});
}

void moveInstBeforeWithDbgInfo(Instruction *Inst, Instruction *Before) {
  // Move the debug info.
  if (Instruction *DbgInst = Inst->getPrevNonDebugInstruction()) {
    DbgInst = DbgInst->getNextNode();
    SmallVector<Instruction *, 8> DbgInsts;
    for (; DbgInst != Inst; DbgInst = DbgInst->getNextNode())
      DbgInsts.push_back(DbgInst);
    for (Instruction *DbgInst : DbgInsts)
      DbgInst->moveBefore(Before);
  }
  // Move the instruction itself.
  Inst->moveBefore(Before);
}

void hoistToEntry(Value *V, BasicBlock *Entry) {
  Instruction *Inst = dyn_cast<Instruction>(V);
  if (!Inst || Inst->getParent() == Entry)
    return;
  Instruction *EntryTerm = Entry->getTerminator();
  for (size_t I = 0; I < Inst->getNumOperands(); I++)
    hoistToEntry(Inst->getOperand(I), Entry);
  if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
    // Also hoist all stores to SrcPtr to the entry block. This is a hack
    // that's not generally correct, but whatever.
    Value *SrcPtr = LI->getPointerOperand();
    Function *F = LI->getFunction();
    SmallVector<StoreInst *, 1> Stores;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (StoreInst *SI = dyn_cast<StoreInst>(&*I))
        if (SI->getPointerOperand() == SrcPtr)
          Stores.push_back(SI);
    }
    for (StoreInst *SI : Stores)
      hoistToEntry(SI, Entry);
  }
  moveInstBeforeWithDbgInfo(Inst, EntryTerm);
}

} // namespace

void CudaLoop::fixReducersInKernel(Function *KernelF, Value *ThreadIdx,
                                   Value *BlockDim, ValueToValueMapTy &VMap) {
  struct ReductionVarInfo {
    Value *LocalPtr; // Holds the thread-local copy of the reduction variable
    size_t Size;
    Function *IdFn;
    Function *MergeFn;
    Type *Type;
    SmallSet<Instruction *, 8> HyperLookups;
    GlobalVariable *SharedPtr;
  };

  ValueMap<Value *, ReductionVarInfo> ReducedVars;
  // Identify all reduction variables in the kernel.
  for (BasicBlock &BB : *KernelF) {
    for (Instruction &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (isTapirIntrinsic(Intrinsic::hyper_lookup, &I)) {
          Value *Ptr = Call->getOperand(0);
          size_t Size = 0;
          if (auto *ConstSize = dyn_cast<ConstantInt>(Call->getOperand(1))) {
            Size = ConstSize->getZExtValue();
          }
          Function *IdFn = dyn_cast<Function>(Call->getOperand(2));
          Function *MergeFn = dyn_cast<Function>(Call->getOperand(3));
          assert(Size > 0 && IdFn && MergeFn &&
                 "cuabi: reduction variable has invalid size, identity "
                 "function, or merge function");
          if (auto It = ReducedVars.find(Ptr); It != ReducedVars.end()) {
            ReductionVarInfo &Record = It->second;
            assert(Size == Record.Size && IdFn == Record.IdFn &&
                   MergeFn == Record.MergeFn &&
                   "cuabi: reduction variable has inconsistent "
                   "size, identity function, or merge function");
          } else {
            Type *ReducerType;
            if (Size <= 8) {
              // Use in-register integers for better performance.
              ReducerType =
                  Type::getIntNTy(KernelF->getContext(), bit_ceil(Size) * 8);
            } else {
              // Use array type for larger types. This is always correct.
              // Backend may of course optimize it into registers
              // Use int32 instead of int8 because that's the native type for
              // CUDA
              size_t NumI32s = (Size + sizeof(int32_t) - 1) / sizeof(int32_t);
              ReducerType = ArrayType::get(
                  Type::getInt32Ty(KernelF->getContext()), NumI32s);
            }
            ReductionVarInfo Record = {nullptr,     Size, IdFn,   MergeFn,
                                       ReducerType, {},   nullptr};
            ReducedVars[Ptr] = Record;
          }
          ReducedVars[Ptr].HyperLookups.insert(Call);
        }
      }
    }
  }

  if (ReducedVars.empty()) {
    LLVM_DEBUG(dbgs() << "cuabi: no reduction variables found in kernel\n");
    return;
  }

  for (auto [Ptr, Info] : ReducedVars) {
    // What a hack, should have done it in preProcessTapirLoop but whatever.
    const auto VMapReverseLookup = [&VMap](Value *Target) -> Value * {
      for (auto [K, V] : VMap) {
        if (V == Target)
          return const_cast<Value *>(K);
      }
      return nullptr;
    };
    Value *HostPtr = VMapReverseLookup(Ptr);
    Function *HostIdFn = dyn_cast<Function>(VMapReverseLookup(Info.IdFn));
    Function *HostMergeFn = dyn_cast<Function>(VMapReverseLookup(Info.MergeFn));
    ReducerInputs[HostPtr] = {Info.Size, HostIdFn, HostMergeFn};
  }

  BasicBlock *Entry = &KernelF->getEntryBlock();
  BasicBlock *Exit = getUniqueExitBlock(KernelF);

  // To fix "inlinable function call in a function with debug info must have a
  // !dbg location" error, we need to attach Id and Merge calls with debug
  // locations in this function. So get the first and last debug locations in
  // the kernel. Id call inserted at the entry block is given the first debug
  // location, and the merge calls inserted at the exit block are given the last
  // debug location.
  const auto FirstAndLastLoc = getFirstAndLastDebugLoc(KernelF);
  //  Insert alloca instructions to create thread-local copies for all reduction
  //  variables.
  {
    IRBuilder<> B(Entry->getTerminator());
    for (auto [Ptr, Info] : ReducedVars) {
      // Cilk reducers are type-erased, so we need to come up with a reasonable
      // type.
      Info.LocalPtr = B.CreateAlloca(Info.Type);
      // Now insert a call to the identity function pointer to initialize the
      // local copy.
      CallInst *IdCall = B.CreateCall(Info.IdFn, {Info.LocalPtr});
      IdCall->setDebugLoc(FirstAndLastLoc.first);
      // Also create a shared memory array for the function for block-wide
      // reduction. Use 32 because CUDA supports up to 32 warps
      ArrayType *ArrayTy = ArrayType::get(Info.Type, MaxNumWarps);
      std::string ArrayName =
          KernelF->getName().str() + ".blk_red_array." + Ptr->getName().str();
      GlobalVariable *Array = new GlobalVariable(
          KernelModule, ArrayTy, false, GlobalValue::InternalLinkage,
          UndefValue::get(ArrayTy), ArrayName, nullptr,
          GlobalVariable::NotThreadLocal, 3);
      Info.SharedPtr = Array;
    }
  }
  // Replace all uses of the reduction variables with the thread-local copies.
  for (auto [Ptr, Info] : ReducedVars) {
    for (Instruction *HyperLookup : Info.HyperLookups) {
      HyperLookup->replaceAllUsesWith(Info.LocalPtr);
      HyperLookup->eraseFromParent();
    }
  }
  // Now generate reduction code at exit block
  IRBuilder<> B(Exit);
  // Remove the existing terminator for now
  Exit->getTerminator()->eraseFromParent();
  Type *Int32Ty = Type::getInt32Ty(KernelF->getContext());
  Value *LaneIdx =
      B.CreateAnd(ThreadIdx, ConstantInt::get(Int32Ty, WarpSize - 1));

  SmallVector<std::vector<std::pair<Value *, ReductionVarInfo>>, 8>
      ReductionTrees;
  if (DontFuseReducers) {
    for (auto [Ptr, Info] : ReducedVars) {
      ReductionTrees.emplace_back();
      ReductionTrees.back().emplace_back(Ptr, Info);
    }
  } else {
    ReductionTrees.emplace_back();
    for (auto [Ptr, Info] : ReducedVars)
      ReductionTrees.back().emplace_back(Ptr, Info);
  }

  for (const auto &CurReducedVars : ReductionTrees) {
    // Let's do a warp-wide reduction first
    const auto TreeReduce = [&](std::string Prefix, size_t Size) {
      const size_t Mask = (1ULL << Size) - 1;
      for (size_t Delta = 1; Delta < Size; Delta *= 2) {
        std::string LoopName = Prefix + "_reduce_" + std::to_string(Delta);
        for (auto &[Ptr, Info] : CurReducedVars) {
          AllocaInst *DownAlloca = B.CreateAlloca(Info.Type);
          // Do a warp shuffle down
          Constant *MaskArg = ConstantInt::get(Int32Ty, Mask);
          Constant *DeltaArg = ConstantInt::get(Int32Ty, Delta);
          Constant *WarpSizeArg = ConstantInt::get(Int32Ty, WarpSize - 1);
          const auto WarpShuffleDown = [&](Value *V) {
            return B.CreateIntrinsic(Intrinsic::nvvm_shfl_sync_down_i32, {},
                                     {MaskArg, V, DeltaArg, WarpSizeArg});
          };
          if (Info.Type->isIntegerTy()) {
            // Use warp shuffle intrinsic
            Value *DownResult = nullptr;
            Value *LocalValue = B.CreateLoad(Info.Type, Info.LocalPtr);
            dbgs() << "cuabi: warp shuffle down " << *LocalValue
                   << " with Info.Size = " << Info.Size << "\n";
            if (Info.Size <= 4) {
              Value *ToShuffle =
                  B.CreateCast(Instruction::CastOps::ZExt, LocalValue, Int32Ty);
              DownResult = B.CreateTrunc(WarpShuffleDown(ToShuffle), Info.Type);
            } else {
              assert(bit_ceil(Info.Size) == 8);
              Value *ToShuffleLow = B.CreateTrunc(LocalValue, Int32Ty);
              Value *ToShuffleHigh = B.CreateTrunc(
                  B.CreateLShr(LocalValue, ConstantInt::get(Info.Type, 32)),
                  Int32Ty);
              Value *ShuffledLow =
                  B.CreateCast(Instruction::CastOps::ZExt,
                               WarpShuffleDown(ToShuffleLow), Info.Type);
              Value *ShuffledHigh =
                  B.CreateCast(Instruction::CastOps::ZExt,
                               WarpShuffleDown(ToShuffleHigh), Info.Type);
              DownResult = B.CreateOr(
                  B.CreateShl(ShuffledHigh, ConstantInt::get(Info.Type, 32)),
                  ShuffledLow);
            }
            B.CreateStore(DownResult, DownAlloca);
          } else {
            assert(Info.Type->isArrayTy() &&
                   Info.Type->getArrayElementType()->isIntegerTy(32));
            // Reference:
            // https://github.com/NVIDIA/cub/blob/0fc3c3701632a4be906765b73be20a9ad0da603d/cub/util_ptx.cuh#L615
            for (size_t Idx = 0; Idx < Info.Type->getArrayNumElements();
                 Idx++) {
              Value *LocalWord =
                  B.CreateLoad(Int32Ty, B.CreateConstInBoundsGEP2_32(
                                            Info.Type, Info.LocalPtr, 0, Idx));
              Value *Shuffled = WarpShuffleDown(LocalWord);
              B.CreateStore(Shuffled, B.CreateConstInBoundsGEP2_32(
                                          Info.Type, DownAlloca, 0, Idx));
            }
          }
          // Brent-Kung reduction -- theoretically more work efficient, but not
          // in practice due to warp divergence. So we use Kogge-Stone instead.
          // Value *Cond = B.CreateICmpEQ(
          //     B.CreateAnd(LaneIdx, ConstantInt::get(Int32Ty, 2 * Delta - 1)),
          //     ConstantInt::get(Int32Ty, 0));
          // Kogge-Stone reduction
          Value *Cond = B.CreateICmpULT(
              B.CreateAdd(LaneIdx, ConstantInt::get(Int32Ty, Delta)),
              ConstantInt::get(Int32Ty, Size));
          std::string ThenName =
              LoopName + "." + Ptr->getName().str() + ".then";
          BasicBlock *ThenBB =
              BasicBlock::Create(KernelF->getContext(), ThenName, KernelF);
          std::string MergeName =
              LoopName + "." + Ptr->getName().str() + ".merge";
          BasicBlock *MergeBB =
              BasicBlock::Create(KernelF->getContext(), MergeName, KernelF);
          B.CreateCondBr(Cond, ThenBB, MergeBB);
          B.SetInsertPoint(ThenBB);
          {
            // Call the merge function
            CallInst *MergeCall =
                B.CreateCall(Info.MergeFn, {Info.LocalPtr, DownAlloca});
            MergeCall->setDebugLoc(FirstAndLastLoc.second);
          }
          B.CreateBr(MergeBB);
          B.SetInsertPoint(MergeBB);
        }
      }
    };
    TreeReduce("warp", WarpSize);

    // Now, if threadIdx % 32 == 0, store the local copy to shared memory
    Value *CondWarpLeader = B.CreateICmpEQ(
        B.CreateAnd(ThreadIdx, ConstantInt::get(Int32Ty, WarpSize - 1)),
        ConstantInt::get(Int32Ty, 0));
    BasicBlock *StoreBB = BasicBlock::Create(
        KernelF->getContext(), "blk_reduce_initial_store", KernelF);
    BasicBlock *AfterStoreBB = BasicBlock::Create(
        KernelF->getContext(), "blk_reduce_after_store", KernelF);
    B.CreateCondBr(CondWarpLeader, StoreBB, AfterStoreBB);
    B.SetInsertPoint(StoreBB);
    {
      // WarpIdx = threadIdx / WarpSize
      Value *WarpIdx =
          B.CreateUDiv(ThreadIdx, ConstantInt::get(Int32Ty, WarpSize));
      // SharedPtr[WarpIdx] = *LocalPtr
      for (auto &[Ptr, Info] : CurReducedVars) {
        if (Info.Type->isIntegerTy()) {
          Value *StorePtr =
              B.CreateInBoundsGEP(Info.Type, Info.SharedPtr, WarpIdx);
          B.CreateStore(B.CreateLoad(Info.Type, Info.LocalPtr), StorePtr);
        } else {
          Type *WarpArrayTy = ArrayType::get(
              Type::getInt32Ty(KernelF->getContext()), MaxNumWarps);
          for (size_t Idx = 0; Idx < Info.Type->getArrayNumElements(); Idx++) {
            Value *LocalWord = B.CreateLoad(
                Int32Ty,
                B.CreateConstInBoundsGEP2_32(Info.Type, Info.LocalPtr, 0, Idx));
            // Actually write to shared[idx][warpIdx] instead of
            // shared[warpIdx][idx] So subsequent read is free of bank
            // conflicts. We don't have write bank conflicts either way
            // because only one thread per warp does this write.
            Value *StorePtr =
                B.CreateInBoundsGEP(WarpArrayTy, Info.SharedPtr,
                                    {ConstantInt::get(Int32Ty, Idx), WarpIdx});
            B.CreateStore(LocalWord, StorePtr);
          }
        }
      }
      B.CreateBr(AfterStoreBB);
    }
    B.SetInsertPoint(AfterStoreBB);
    B.CreateIntrinsic(Intrinsic::nvvm_bar_sync, {},
                      {ConstantInt::get(Int32Ty, 0)});

    // Now do a block-wide reduction from shared memory
    assert(MaxNumWarps <= WarpSize &&
           "Block reduction can't be done in a warp");
    Value *CondFirstWarp =
        B.CreateICmpULT(ThreadIdx, ConstantInt::get(Int32Ty, MaxNumWarps));
    BasicBlock *FirstWarpBB = BasicBlock::Create(
        KernelF->getContext(), "blk_reduce_first_warp", KernelF);
    BasicBlock *OtherWarpBB = BasicBlock::Create(
        KernelF->getContext(), "blk_reduce_other_warp", KernelF);
    B.CreateCondBr(CondFirstWarp, FirstWarpBB, OtherWarpBB);

    B.SetInsertPoint(FirstWarpBB);
    {
      Value *NumWarps =
          B.CreateUDiv(BlockDim, ConstantInt::get(Int32Ty, WarpSize));
      // Each thread will load SharedPtr[threadIdx] into its local copy
      // But if the block is small, SharedPtr[threadIdx] may not have been
      // written to and contains junk. In that case, that thread should
      // initialize its local copy to identity.
      Value *CondHasData = B.CreateICmpULT(ThreadIdx, NumWarps);
      BasicBlock *LoadFromSHMemBB = BasicBlock::Create(
          KernelF->getContext(), "blk_reduce_load_from_shmem", KernelF);
      BasicBlock *InitIdBB = BasicBlock::Create(KernelF->getContext(),
                                                "blk_reduce_init_id", KernelF);
      BasicBlock *StartReduceBB = BasicBlock::Create(
          KernelF->getContext(), "blk_reduce_start", KernelF);
      B.CreateCondBr(CondHasData, LoadFromSHMemBB, InitIdBB);
      B.SetInsertPoint(LoadFromSHMemBB);
      {
        // *LocalPtr = SharedPtr[threadIdx]
        for (auto &[Ptr, Info] : CurReducedVars) {
          if (Info.Type->isIntegerTy()) {
            Value *SHMemPtr =
                B.CreateInBoundsGEP(Info.Type, Info.SharedPtr, ThreadIdx);
            B.CreateStore(B.CreateLoad(Info.Type, SHMemPtr), Info.LocalPtr);
          } else {
            Type *WarpArrayTy = ArrayType::get(
                Type::getInt32Ty(KernelF->getContext()), MaxNumWarps);
            for (size_t Idx = 0; Idx < Info.Type->getArrayNumElements();
                 Idx++) {
              Value *SHMemWord = B.CreateInBoundsGEP(
                  WarpArrayTy, Info.SharedPtr,
                  {ConstantInt::get(Int32Ty, Idx), ThreadIdx});
              B.CreateStore(B.CreateLoad(Int32Ty, SHMemWord),
                            B.CreateConstInBoundsGEP2_32(
                                Info.Type, Info.LocalPtr, 0, Idx));
            }
          }
        }
        B.CreateBr(StartReduceBB);
      }
      B.SetInsertPoint(InitIdBB);
      {
        // Identity(*LocalPtr)
        for (auto &[Ptr, Info] : CurReducedVars) {
          CallInst *IdCall = B.CreateCall(Info.IdFn, {Info.LocalPtr});
          IdCall->setDebugLoc(FirstAndLastLoc.second);
        }
        B.CreateBr(StartReduceBB);
      }
      B.SetInsertPoint(StartReduceBB);
      // Load the shared memory array
      TreeReduce("blk", MaxNumWarps);
      B.CreateBr(OtherWarpBB);
    }
    B.SetInsertPoint(OtherWarpBB);
    Value *CondBlockLeader =
        B.CreateICmpEQ(ThreadIdx, ConstantInt::get(Int32Ty, 0));
    BasicBlock *WriteToGlobalBB = BasicBlock::Create(
        KernelF->getContext(), "blk_reduce_write_to_global", KernelF);
    BasicBlock *ExitBB =
        BasicBlock::Create(KernelF->getContext(), "blk_reduce_exit", KernelF);
    B.CreateCondBr(CondBlockLeader, WriteToGlobalBB, ExitBB);
    B.SetInsertPoint(WriteToGlobalBB);
    {
      for (auto &[Ptr, Info] : CurReducedVars) {
        if (Info.Type->isIntegerTy()) {
          // Write a CAS loop to write the local copy to global memory
          Value *OldPtr = B.CreateAlloca(Info.Type);
          Value *NewPtr = B.CreateAlloca(Info.Type);
          BasicBlock *LoopBB = BasicBlock::Create(
              KernelF->getContext(),
              "blk_reduce_write_loop_" + Ptr->getName().str(), KernelF);
          BasicBlock *DoneBB = BasicBlock::Create(
              KernelF->getContext(),
              "blk_reduce_write_done_" + Ptr->getName().str(), KernelF);
          LoadInst *LoadedOldVal = B.CreateLoad(Info.Type, Ptr);
          // LoadedOldVal->setAtomic(AtomicOrdering::SequentiallyConsistent);
          BasicBlock *PrevBlock = B.GetInsertBlock();
          B.CreateBr(LoopBB);
          B.SetInsertPoint(LoopBB);
          {
            PHINode *OldVal = B.CreatePHI(Info.Type, 2);
            B.CreateStore(OldVal, OldPtr);
            B.CreateStore(B.CreateLoad(Info.Type, Info.LocalPtr), NewPtr);
            CallInst *MergeCall = B.CreateCall(Info.MergeFn, {NewPtr, OldPtr});
            MergeCall->setDebugLoc(FirstAndLastLoc.second);
            Value *CASResult = B.CreateAtomicCmpXchg(
                Ptr, OldVal, B.CreateLoad(Info.Type, NewPtr),
                MaybeAlign(Info.Size), AtomicOrdering::SequentiallyConsistent,
                AtomicOrdering::SequentiallyConsistent);
            Value *CASOldVal = B.CreateExtractValue(CASResult, 0);
            Value *CASSuccess = B.CreateExtractValue(CASResult, 1);
            OldVal->addIncoming(CASOldVal, LoopBB);
            OldVal->addIncoming(LoadedOldVal, PrevBlock);
            B.CreateCondBr(CASSuccess, DoneBB, LoopBB);
          }
          B.SetInsertPoint(DoneBB);
        } else {
          // If aggregate type, it's tricky. Atomic CAS will not work. But
          // host-side will allocate a mutex right after the global view.
          Value *MutexPtr = B.CreateConstGEP1_32(
              Int32Ty, Ptr, Info.Type->getArrayNumElements());
          BasicBlock *LoopBB = BasicBlock::Create(
              KernelF->getContext(),
              "blk_reduce_mutex_loop_" + Ptr->getName().str(), KernelF);
          BasicBlock *DoneBB = BasicBlock::Create(
              KernelF->getContext(),
              "blk_reduce_mutex_done_" + Ptr->getName().str(), KernelF);
          B.CreateBr(LoopBB);
          B.SetInsertPoint(LoopBB);
          { // Loop to acquire the mutex
            Value *CASResult = B.CreateAtomicCmpXchg(
                MutexPtr, ConstantInt::get(Int32Ty, 0),
                ConstantInt::get(Int32Ty, 1), MaybeAlign(4),
                AtomicOrdering::SequentiallyConsistent,
                AtomicOrdering::SequentiallyConsistent);
            Value *CASSuccess = B.CreateExtractValue(CASResult, 1);
            B.CreateCondBr(CASSuccess, DoneBB, LoopBB);
          }
          B.SetInsertPoint(DoneBB);
          B.CreateIntrinsic(Intrinsic::nvvm_membar_gl, {}, {});
          CallInst *MergeCall =
              B.CreateCall(Info.MergeFn, {Ptr, Info.LocalPtr});
          MergeCall->setDebugLoc(FirstAndLastLoc.second);
          B.CreateIntrinsic(Intrinsic::nvvm_membar_gl, {}, {});
          // Release the mutex
          B.CreateStore(ConstantInt::get(Int32Ty, 0), MutexPtr);
        }
      }
      B.CreateBr(ExitBB);
    }
    B.SetInsertPoint(ExitBB);
  }
  B.CreateRetVoid();
}

void CudaLoop::fixScannersInKernel(Function *KernelF, Value *TripCount,
                                   ValueToValueMapTy &VMap) {
  auto *Int32Ty = Type::getInt32Ty(KernelF->getContext());
  struct ScannerInfo {
    size_t Size;
    Type *Type;
    CallInst *Call;
  };
  SmallVector<ScannerInfo, 8> Scanners;
  for (inst_iterator I = inst_begin(KernelF), E = inst_end(KernelF); I != E;
       ++I) {
    if (auto *Call = dyn_cast<CallInst>(&*I)) {
      if (!Call->getCalledFunction() ||
          Call->getCalledFunction()->getName() != "__kitcuda_get_scan_view")
        continue;
      size_t Size = 0;
      if (auto *ConstSize = dyn_cast<ConstantInt>(Call->getArgOperand(0))) {
        Size = ConstSize->getZExtValue();
      }
      assert(Size > 0 && Size % sizeof(int32_t) == 0 &&
             "cuabi: scanner variable has invalid size");
      size_t NumI32s = Size / sizeof(int32_t);
      Type *Ty =
          NumI32s == 1
              ? static_cast<Type *>(Int32Ty) // Does it improve performance?
              : ArrayType::get(Int32Ty, NumI32s);
      Scanners.push_back({Size, Ty, Call});
    }
  }

  if (Scanners.empty())
    return;

  const auto VMapReverseLookup = [&VMap](Value *Target) -> Value * {
    for (auto [K, V] : VMap) {
      if (V == Target)
        return const_cast<Value *>(K);
    }
    return nullptr;
  };

  BasicBlock *Entry = &KernelF->getEntryBlock();
  BasicBlock *Exit = getUniqueExitBlock(KernelF);
  const auto FirstAndLastLoc = getFirstAndLastDebugLoc(KernelF);
  IRBuilder<> EntryBuilder(Entry->getTerminator());
  IRBuilder<> ExitBuilder(Exit->getTerminator());

  auto *PtrTy = PointerType::get(KernelModule.getContext(), 0);
  auto *Int64Ty = Type::getInt64Ty(KernelModule.getContext());
  auto *VoidTy = Type::getVoidTy(KernelModule.getContext());
  FunctionCallee KitCudaScanFn = KernelModule.getOrInsertFunction(
      "__kitcuda_scan", /* returns */ VoidTy,
      /* view */ PtrTy, /* temp_1 */ PtrTy, /* temp_2 */ PtrTy,
      /* temp_3 */ PtrTy, /* shmem */ PtrTy,
      /* aggregate */ PtrTy, /* inclusive_prefix */ PtrTy,
      /* scan_state */ PtrTy, /* size */ Int64Ty,
      /* result */ PtrTy, /* n */ Int64Ty, /* identity */ PtrTy,
      /* reduce */ PtrTy);

  for (ScannerInfo &SI : Scanners) {
    // Create Allocas
    Value *View = EntryBuilder.CreateAlloca(SI.Type);
    Value *Temp1 = EntryBuilder.CreateAlloca(SI.Type);
    Value *Temp2 = EntryBuilder.CreateAlloca(SI.Type);
    Value *Temp3 = EntryBuilder.CreateAlloca(SI.Type);

    Value *Aggregate = SI.Call->getArgOperand(1);
    Value *InclusivePrefix = SI.Call->getArgOperand(2);
    Value *ScanState = SI.Call->getArgOperand(3);
    Value *Result = SI.Call->getArgOperand(4);
    Value *IdFn = SI.Call->getArgOperand(5);
    Value *ReduceFn = SI.Call->getArgOperand(6);

    ArrayType *ArrayTy =
        ArrayType::get(Int32Ty, MaxNumWarps * (SI.Size / sizeof(int32_t)));
    std::string ArrayName = KernelF->getName().str() + ".blk_scan_array." +
                            SI.Call->getArgOperand(0)->getName().str();
    GlobalVariable *Array = new GlobalVariable(
        KernelModule, ArrayTy, false, GlobalValue::InternalLinkage,
        UndefValue::get(ArrayTy), ArrayName, nullptr,
        GlobalVariable::NotThreadLocal, 3);
    Value *Shmem = ExitBuilder.CreateAddrSpaceCast(Array, PtrTy);
    CallInst *ScanCall = ExitBuilder.CreateCall(
        KitCudaScanFn,
        {View, Temp1, Temp2, Temp3, Shmem, Aggregate, InclusivePrefix,
         ScanState, ConstantInt::get(Int64Ty, SI.Size), Result, TripCount, IdFn,
         ReduceFn});
    ScanCall->setDebugLoc(FirstAndLastLoc.second);
    CallInst *IdCall =
        EntryBuilder.CreateCall(dyn_cast<Function>(IdFn), {View});
    IdCall->setDebugLoc(FirstAndLastLoc.first);

    SI.Call->replaceAllUsesWith(View);
    SI.Call->eraseFromParent();

    ScannerInputs.push_back({SI.Size, VMapReverseLookup(Aggregate),
                             VMapReverseLookup(InclusivePrefix),
                             VMapReverseLookup(ScanState)});
  }
}

Function *CudaLoop::resolveLibDeviceFunction(Function *Fn, bool enableFast) {
  std::unique_ptr<Module> &LDM = TTarget->getLibDeviceModule();
  const std::string NVPrefix = "__nv_";

  // Handle special cases where code generation can be a bit more
  // complex; e.g., printf().
  if (Fn->getName() == "printf" || Fn->getName() == "fprintf") {
    report_fatal_error("cuabi: printf is currently unsupported "
                       "in parallel loops... :-(\n");
  }

  std::string FnName = "";
  if (Fn->isIntrinsic()) {

    if (enableFast)
      FnName = "fast_";

    if (Fn->getName().str().compare(0, 9, "llvm.nvvm") == 0)
      return nullptr; // backend can handle these...
    else if (Fn->getName() == "llvm.cos.f32")
      FnName += "cosf";
    else if (Fn->getName() == "llvm.cos.f64")
      FnName += "cos";
    else if (Fn->getName() == "llvm.sin.f32")
      FnName += "sinf";
    else if (Fn->getName() == "llvm.sin.f64")
      FnName += "sin";
    else if (Fn->getName() == "llvm.tan.f32")
      FnName += "tanf";
    else if (Fn->getName() == "llvm.tan.f64")
      FnName += "tan";
    else if (Fn->getName() == "llvm.exp.f64")
      FnName += "exp";
    else if (Fn->getName() == "llvm.expf.f32")
      FnName += "expf";
    else {
      // errs() << "cuabi: transforming intrinsic call " << Fn->getName() <<
      // "()\n"; report_fatal_error("cuabi: no transform for llvm intrinsic!");
      return nullptr;
    }
  } else {
    if (Fn->getName() == "__sqrtf_finite") {
      FnName = "llvm.nvvm.sqrt.approx.ftz.f";
      // errs() << "\t mapping to " << FnName << "\n";
    } else if (Fn->getName() == "__powf_finite")
      FnName = "fast_powf";
    else if (Fn->getName() == "__fmodf_finite")
      FnName = "modff";
    else if (Fn->getName() == "expf") {
      if (enableFast)
        FnName = "__nv_fast_expf";
      else
        FnName = "__nv_expf";
      errs() << "call for exp: " << FnName << "().\n";
    }
  }

  FnName = Fn->getName().str();
  if (Function *KF = KernelModule.getFunction(NVPrefix + FnName)) {
    LLVM_DEBUG(dbgs() << "\t\tfound existing device function '" << KF->getName()
                      << "'.\n");
    return KF;
  }

  for (auto &DF : *LDM) {
    std::string DFName = DF.getName().str();
    auto Match =
        std::mismatch(NVPrefix.begin(), NVPrefix.end(), DFName.begin());
    auto BaseName = DFName.substr(Match.second - DFName.begin());
    if (BaseName == FnName) {
      LLVM_DEBUG(dbgs() << "Found libdevice function: '" << DF.getName()
                        << "' to resolve function '" << FnName << "'.\n");
      return &DF;
    }
  }
  return nullptr;
}

void CudaLoop::transformForPTX(Function &F) {

  // LLVM_DEBUG(dbgs() << "Transforming function '" << F.getName() << "' "
  //                   << "in preparation for PTX generation.\n");

  // PTX doesn't like .<n> global names, rename them to
  // replace the '.' with an underscore, '_'.
  for (GlobalVariable &G : KernelModule.globals()) {
    auto name = G.getName().str();
    std::replace(name.begin(), name.end(), '.', '_');
    G.setName(name);
  }


  // We now need to walk the kernel (outlined loop) and look for
  // unresolved function calls.  In particular we need to check
  // to see if they can be resolved via CUDA's libdevice.
  // LLVM_DEBUG(
  //    dbgs() << "cuabi: search for unresolved calls in outlined kernel...\n");
  std::list<CallInst *> Replaced;
  bool enableFast;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      if (FPMathOperator *FPO = dyn_cast<FPMathOperator>(CI)) {
        if (FPO->isFast()) {
          enableFast = true;
        } else {
          // LLVM_DEBUG(dbgs() << " [std/full precision]\n");
          enableFast = false;
        }
      }

      Function *CF = CI->getCalledFunction();
      if (CF && CF->size() == 0) {
        Function *DF = resolveLibDeviceFunction(CF, enableFast);
        if (DF != nullptr) {
          CallInst *NCI = dyn_cast<CallInst>(CI->clone());
          NCI->insertBefore(CI);
          NCI->setCalledFunction(DF);
          CI->replaceAllUsesWith(NCI);
          Replaced.push_back(CI);
        }
      }
    }
  }

  for (auto CI : Replaced)
    CI->eraseFromParent();

  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> KernelIRFile;
    SmallString<255> IRFileName(Twine(F.getName()).str() + "-ptx");
    sys::path::replace_extension(IRFileName, ".ll");
    KernelIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    KernelModule.print(KernelIRFile->os(), nullptr);
    KernelIRFile->keep();
  }
}

void CudaLoop::remapData(ValueToValueMapTy &VMap) {
  for (auto &V : OrderedInputs) {
    if (auto MappedV = VMap[V]) {
      V = MappedV;
    }
  }
  DenseMap<const Value *, ReducerOutlineLoopCallInfo> NewReducerInputs;
  for (auto [V, Info] : ReducerInputs) {
    if (auto MappedV = VMap[V]) {
      NewReducerInputs[MappedV] = Info;
    } else {
      NewReducerInputs[V] = Info;
    }
  }
  ReducerInputs = std::move(NewReducerInputs);
  for (ScannerOutlineLoopCallInfo &SOI : ScannerInputs) {
    if (auto MappedV = VMap[SOI.Aggregate])
      SOI.Aggregate = MappedV;
    if (auto MappedV = VMap[SOI.InclusivePrefix])
      SOI.InclusivePrefix = MappedV;
    if (auto MappedV = VMap[SOI.ScanState])
      SOI.ScanState = MappedV;
  }
  // Remap SyncRegList
  CudaABI::SyncRegionListTy NewSyncRegList;
  for (auto &SyncReg : TTarget->SyncRegList) {
    if (auto MappedV = VMap[SyncReg]) {
      NewSyncRegList.insert(MappedV);
    } else {
      NewSyncRegList.insert(SyncReg);
    }
  }
  TTarget->SyncRegList = std::move(NewSyncRegList);
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();

  // NOTE: If we are dealing with loop nests with multiple targets
  // (in this case only a CPU-target w/ a nested GPU target is
  // supported) we can end up with multiple calls to the outlined
  // loop (which has been setup for dead code elimination) but can
  // cause invalid IR that trips us up when handling the GPU module
  // code generation. This is a challenge in the Tapir design that
  // was not geared to handle some of the nuances of GPU target
  // transformations (and code gen).  To address this, we need to
  // do some clean up to keep the IR correct (or the verifier will
  // fail on us...).  Specifically, we can no longer depend upon
  // DCE as it runs too late in the GPU transformation process...
  //
  // TODO: This code can be shared between the cuda and hip targets...
  //
  Function *TargetKF = KernelModule.getFunction(KernelName);
  std::list<Instruction *> RemoveList;
  if (TargetKF) {
    LLVM_DEBUG(dbgs() << "\t*- searching for 'dangling' outline calls...\n");
    for (Use &U : TargetKF->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser())) {
        LLVM_DEBUG(dbgs() << "\t\t- marking use for removal.\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }

  for (auto I : RemoveList)
    I->eraseFromParent();

  // Make a pass to prep for PTX code generation...
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForPTX(F);
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");

  // Create two builders -- one inserts code into the entry block
  // (e.g. new "up-front" allocas) and the other is for generating
  // new code into a split BB.
  Function *Parent = TOI.ReplCall->getFunction();
  BasicBlock &EntryBB = Parent->getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB.front());

  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(TOI.ReplCall);
  IRBuilder<> NewBuilder(&NewBB->front());

  // TODO: There is some potential here to share this code across both
  // the hip and cuda transforms...
  LLVM_DEBUG(dbgs() << "\t*- code gen packing of " << OrderedInputs.size()
                    << " kernel args.\n");
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EntryBuilder.CreateAlloca(ArrayTy);
  AllocaInst *CudaStream = EntryBuilder.CreateAlloca(VoidPtrTy);
  EntryBuilder.CreateStore(ConstantPointerNull::get(VoidPtrTy), CudaStream);
  bool StreamAssigned = false;
  unsigned int i = 0;
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  CudaABI::ReducersLaunchInfo LaunchInfo;
  ValueToValueMapTy DeviceReducerViews;

  if (!ReducerInputs.empty()) {
    // Allocate temporary host view for reducers used
    SmallVector<std::pair<const Value *, ReducerOutlineLoopCallInfo>>
        SortedReducerInputs;
    for (auto &[V, Info] : ReducerInputs) {
      SortedReducerInputs.push_back({V, Info});
    }
    // Sort the reducer inputs from biggest to smallest
    std::sort(SortedReducerInputs.begin(), SortedReducerInputs.end(),
              [](const auto &LHS, const auto &RHS) {
                return LHS.second.Size > RHS.second.Size;
              });
    size_t ViewSize = 0;
    for (auto &[V, Info] : SortedReducerInputs) {
      size_t Alignment = std::min(bit_ceil(Info.Size), size_t{8});
      ViewSize = (ViewSize + Alignment - 1) & -Alignment;
      ViewSize += Info.Size;
      if (Info.Size > 8) {
        ViewSize = (ViewSize + 3) & -4;
        // Extra 4 bytes for the mutex
        ViewSize += 4;
      }
    }
    LaunchInfo.ViewSize = ViewSize;

    Type *ViewTy = ArrayType::get(Type::getInt8Ty(Ctx), ViewSize);
    Value *HostViewBasePtr = EntryBuilder.CreateAlloca(ViewTy);
    LaunchInfo.HostViewBasePtr = HostViewBasePtr;
    Value *DeviceViewPtrPtr = EntryBuilder.CreateAlloca(VoidPtrTy);
    LaunchInfo.DeviceViewBasePtrPtr = DeviceViewPtrPtr;
    CallInst *NewSPtr = NewBuilder.CreateCall(
        KitCudaMemAllocAndCopyToDevFn,
        {HostViewBasePtr, ConstantInt::get(Int64Ty, ViewSize), DeviceViewPtrPtr,
         NewBuilder.CreateLoad(VoidPtrTy, CudaStream)});
    if (not StreamAssigned) {
      NewBuilder.CreateStore(NewSPtr, CudaStream);
      StreamAssigned = true;
    }

    Value *DeviceViewBasePtr =
        NewBuilder.CreateLoad(VoidPtrTy, DeviceViewPtrPtr);
    // Now prepend instructions before __kitcuda_mem_alloc_and_copy_to_dev
    IRBuilder<> PreCopyBuilder(NewSPtr);
    ViewSize = 0;
    for (auto &[V, Info] : SortedReducerInputs) {
      size_t Alignment = std::min(bit_ceil(Info.Size), size_t{8});
      ViewSize = (ViewSize + Alignment - 1) & -Alignment;
      Value *HostViewPtr = EntryBuilder.CreateConstInBoundsGEP2_32(
          ViewTy, HostViewBasePtr, 0, ViewSize);
      PreCopyBuilder.CreateCall(Info.IdFn, {HostViewPtr});
      Value *DeviceViewPtr = NewBuilder.CreateConstInBoundsGEP2_32(
          ViewTy, DeviceViewBasePtr, 0, ViewSize);
      LaunchInfo.ReducerInfos.push_back({const_cast<Value *>(V), HostViewPtr,
                                         Info.Size, Info.IdFn, Info.MergeFn});
      DeviceReducerViews[V] = DeviceViewPtr;
      ViewSize += Info.Size;
      if (Info.Size > 8) {
        ViewSize = (ViewSize + 3) & -4;
        // Extra 4 bytes for the mutex
        PreCopyBuilder.CreateStore(ConstantInt::get(Type::getInt32Ty(Ctx), 0),
                                   EntryBuilder.CreateConstInBoundsGEP2_32(
                                       ViewTy, HostViewBasePtr, 0, ViewSize));
        ViewSize += 4;
      }
    }
  }

  ValueToValueMapTy DeviceScannerAuxArrays;
  SmallVector<CallInst *, 1> ScannerAllocCalls;
  if (!ScannerInputs.empty()) {
    Type *VoidTy = Type::getVoidTy(Ctx);
    FunctionCallee KitCudaAllocScanner = M.getOrInsertFunction(
        "__kitcuda_alloc_scanner", VoidTy, VoidPtrTy, VoidPtrTy, Int64Ty,
        VoidPtrTy, Int64Ty, VoidPtrTy, VoidPtrTy, VoidPtrTy);
    Value *NullPtrPlaceholder = ConstantPointerNull::get(VoidPtrTy);
    for (ScannerOutlineLoopCallInfo &SOI : ScannerInputs) {
      Value *AggregatePtr = EntryBuilder.CreateAlloca(VoidPtrTy);
      Value *InclusivePrefixPtr = EntryBuilder.CreateAlloca(VoidPtrTy);
      Value *ScanStatePtr = EntryBuilder.CreateAlloca(VoidPtrTy);
      CallInst *ScannerAllocCall = NewBuilder.CreateCall(
          KitCudaAllocScanner,
          {
              // These four values are placeholders and will be replaced later
              NullPtrPlaceholder,           // Fatbinary
              NullPtrPlaceholder,           // Kernel name
              ConstantInt::get(Int64Ty, 0), // Trip count
              NullPtrPlaceholder,           // InstMix

              ConstantInt::get(Int64Ty, SOI.Size),
              AggregatePtr,
              InclusivePrefixPtr,
              ScanStatePtr,
          });
      ScannerAllocCalls.push_back(ScannerAllocCall);
      DeviceScannerAuxArrays[SOI.Aggregate] =
          NewBuilder.CreateLoad(VoidPtrTy, AggregatePtr);
      DeviceScannerAuxArrays[SOI.InclusivePrefix] =
          NewBuilder.CreateLoad(VoidPtrTy, InclusivePrefixPtr);
      DeviceScannerAuxArrays[SOI.ScanState] =
          NewBuilder.CreateLoad(VoidPtrTy, ScanStatePtr);
    }
  }

  for (Value *V : OrderedInputs) {
    Value *RealV = V;
    bool IsReducerOrScanner = false;
    if (auto ReducerInput = DeviceReducerViews.find(V);
        ReducerInput != DeviceReducerViews.end()) {
      IsReducerOrScanner = true;
      RealV = ReducerInput->second;
    }
    if (auto ScannerInput = DeviceScannerAuxArrays.find(V);
        ScannerInput != DeviceScannerAuxArrays.end()) {
      IsReducerOrScanner = true;
      RealV = ScannerInput->second;
    }

    Value *VP = EntryBuilder.CreateAlloca(V->getType());
    NewBuilder.CreateStore(RealV, VP);
    Value *VoidVPtr = NewBuilder.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr =
        NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    NewBuilder.CreateStore(VoidVPtr, ArgPtr);
    i++;

    if (!IsReducerOrScanner && CodeGenPrefetch &&
        RealV->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "\t\t- code gen prefetch for kernel arg #" << i
                        << "\n");
      Value *VoidPP = NewBuilder.CreateBitCast(RealV, VoidPtrTy);
      Value *SPtr = NewBuilder.CreateLoad(VoidPtrTy, CudaStream);
      Value *NewSPtr =
          NewBuilder.CreateCall(KitCudaMemPrefetchFn, {VoidPP, SPtr});
      if (not StreamAssigned) {
        NewBuilder.CreateStore(NewSPtr, CudaStream);
        StreamAssigned = true;
      }
    }
  }

  // The next step is prep for the actual kernel launch call via
  // the kitsune runtime.  We have to add some extra levels of
  // pointers to match API details, deal with some potential
  // type mismatches, build a dummy pointer for the yet-to-be-created
  // fat binary, etc...
  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr =
      NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, "kern.name");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We place *all* transformed tapir loops from the input module into a
  // single GPU target module.  At this point we can not create a complete
  // fat binary image.  However, we have all the important info for the
  // current loop so we use a 'dummy' (null) fat binary for code gen at
  // this point -- we'll post-process the module to clean this up after
  // we've processed all tapir loops.
  Constant *DummyFBGV =
      tapir::getOrInsertFBGlobal(M, "_cuabi.dummy_fatbin", VoidPtrTy);
  Value *DummyFBPtr = NewBuilder.CreateLoad(VoidPtrTy, DummyFBGV);

  // Deal with type mismatches for the trip count. A difference
  // introduced via the input source details and the runtime's
  // API type signature for the launch.
  Value *TripCount = OrderedInputs[0];
  Value *CastTripCount = nullptr;
  if (TripCount->getType() != Int64Ty) {
    CastTripCount = CastInst::CreateIntegerCast(TripCount, Int64Ty, false);
    NewBuilder.Insert(CastTripCount, "cast.tc");
  } else
    CastTripCount = TripCount;

  // Determine if this kernel has nested tapir loops
  // by checking for the nested parallelism metadata we set earlier
  // (we can't check for detach instructions here because they've been serialized)
  bool HasNestedThreadLoops = false;
  if (Function *KernelFunc = KernelModule.getFunction(KernelName)) {
    if (KernelFunc->getMetadata("cuda.kernel.nested")) {
      HasNestedThreadLoops = true;
      LLVM_DEBUG(dbgs() << "\t*- detected nested thread-parallel loops "
                        << "in kernel launch (from metadata)\n");
    }
  }

  // At this point we need a threads-per-block value for the launch
  // call.  The runtime will determine this value if ThreadsPerBlock
  // is zero but it can also be overridden via kitsune's forall launch
  // attribute.  The catch here is the launch attribute's value for
  // this is flexible and be a computed expression vs. a compile-time
  // constant.  For this first step of creating the kernel launch, we
  // take the path of a runtime configuration vs. an attributed
  // launch.  This will get patched up as needed when we post-process
  // the module and replace the DummyFBPtr (as we will also need to
  // replace the kernel launch call parameter for threads-per-block if
  // an attributed expression is present).  See postProcessModule()'s
  // stage of finalizing the launch calls for details.
  TapirLoopHints Hints(TL.getLoop());
  unsigned ThreadsPerBlock = 0;
  
  // For nested block/thread parallel loops:
  // - TripCount = outer loop iterations = number of blocks to launch
  // - TPB is determined at runtime via CUDA occupancy analysis
  // For single-level loops:
  // - TripCount = total iterations, TPB = 0 (runtime decides)
  if (HasNestedThreadLoops) {
    LLVM_DEBUG(dbgs() << "\t*- nested launch: TPB from occupancy, num_blocks = trip_count\n");
  }
  
  Constant *TPBlockValue =
      ConstantInt::get(Type::getInt32Ty(Ctx), ThreadsPerBlock);

  LLVM_DEBUG(dbgs() << "\tgathering kernel instruction mix....\n");
  tapir::KernelInstMixData InstMix;
  tapir::getKernelInstructionMix(&F, InstMix);
  LLVM_DEBUG(
      dbgs() << "\tinstruction mix:\n"
             << "      memory ops      : " << InstMix.num_memory_ops << "\n"
             << "      flop count      : " << InstMix.num_flops << "\n"
             << "      integer op count: " << InstMix.num_iops << "\n\n");

  Constant *InstructionMix = ConstantStruct::get(
      KernelInstMixTy, ConstantInt::get(Int64Ty, InstMix.num_memory_ops),
      ConstantInt::get(Int64Ty, InstMix.num_flops),
      ConstantInt::get(Int64Ty, InstMix.num_iops));

  AllocaInst *AI = NewBuilder.CreateAlloca(KernelInstMixTy);
  NewBuilder.CreateStore(InstructionMix, AI);

  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  
  Value *KSPtr = NewBuilder.CreateLoad(VoidPtrTy, CudaStream);
  CallInst *LaunchStream;
  
  if (HasNestedThreadLoops) {
    LLVM_DEBUG(dbgs() << "\t\t nested launch: num_blocks = " << *CastTripCount << "\n");
    LaunchStream = NewBuilder.CreateCall(
        KitCudaLaunchNestedFn, {DummyFBPtr, KNameParam, argsPtr, CastTripCount,
                                AI, KSPtr});
  } else {
    LLVM_DEBUG(dbgs() << "\t\t trip count: " << *CastTripCount << "\n");
    LLVM_DEBUG(dbgs() << "\t\t threads per block: " << ThreadsPerBlock 
                      << " (0=runtime decides)\n");
    LaunchStream = NewBuilder.CreateCall(
        KitCudaLaunchFn, {DummyFBPtr, KNameParam, argsPtr, CastTripCount,
                          TPBlockValue, AI, KSPtr});
  }
  // if (not StreamAssigned)
  NewBuilder.CreateStore(LaunchStream, CudaStream);
  LLVM_DEBUG(dbgs() << "\t\t+- registering launch stream:\n"
                    << "\t\t\tcall: " << *LaunchStream << "\n"
                    << "\t\t\tstream: " << *CudaStream << "\n");
  TTarget->registerLaunchStream(LaunchStream, CudaStream);
  if (!ReducerInputs.empty()) {
    TTarget->registerReducerLaunchInfo(LaunchStream, LaunchInfo);
  }

  LLVM_DEBUG(dbgs() << "\t\t+- deferred sync: " << Hints.getDeferredSync()
                    << "\n");

  if (!Hints.getDeferredSync() || !ReducerInputs.empty() ||
      !ScannerInputs.empty()) {
    // Only generate sync call when the loop is not marked default sync or if
    // it's a reduction
    Type *VoidTy = Type::getVoidTy(Ctx);
    FunctionCallee KitCudaSyncFn = M.getOrInsertFunction(
        "__kitcuda_sync_thread_stream", VoidTy, VoidPtrTy);
    NewBuilder.CreateCall(KitCudaSyncFn, {LaunchStream});
    if (!ScannerInputs.empty()) {
      FunctionCallee KitCudaFreeScannerFn = M.getOrInsertFunction(
          "__kitcuda_free_scanner", VoidTy, VoidPtrTy, VoidPtrTy, VoidPtrTy);
      for (ScannerOutlineLoopCallInfo &SOI : ScannerInputs) {
        NewBuilder.CreateCall(KitCudaFreeScannerFn,
                              {DeviceScannerAuxArrays[SOI.Aggregate],
                               DeviceScannerAuxArrays[SOI.InclusivePrefix],
                               DeviceScannerAuxArrays[SOI.ScanState]});
      }
      for (CallInst *ScannerAllocCall : ScannerAllocCalls) {
        ScannerAllocCall->setArgOperand(0, DummyFBPtr);
        ScannerAllocCall->setArgOperand(1, KNameParam);
        ScannerAllocCall->setArgOperand(2, CastTripCount);
        ScannerAllocCall->setArgOperand(3, AI);
      }
      // Move AI to befoer teh first ScannerAllocCall
      AI->moveBefore(ScannerAllocCalls.front());
    }
  }
  TOI.ReplCall->eraseFromParent();

  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &M)
    : TapirTarget(M),
      KernelModule(Twine(CUABI_PREFIX + sys::path::filename(M.getName())).str(),
                   M.getContext()) {

  // A helping hand for invocation of the transform from a JIT-driven
  // environment where it can be a bit painful to get to
  std::optional<std::string> envTarget = sys::Process::GetEnv("CUDAABI_TARGET");
  if (envTarget) {
    LLVM_DEBUG(dbgs() << "cuabi: target set via environment '"
                      << envTarget.value() << "'.\n");
    GPUArch.setInitialValue(envTarget.value());
  }

  std::optional<std::string> ThreadsPBVar =
      sys::Process::GetEnv("CUDABI_DEFAULT_THREADS_PER_BLOCK");
  if (ThreadsPBVar) {
    MaxThreadsPerBlock = std::stoi(ThreadsPBVar.value());
    if (MaxThreadsPerBlock <= 0 ||
        MaxThreadsPerBlock > CUDAABI_MAX_THREADS_PER_BLOCK)
      report_fatal_error(
          "CUDAABI_DEFAULT_THREADS_PER_BLOCK must be > 0 && <= 1024!");
  }
  LLVM_DEBUG(dbgs() << "cuabi: setting max threads per block: "
                    << MaxThreadsPerBlock << "\n");

  LLVM_DEBUG(dbgs() << "cuabi: creating tapir target for module '"
                    << M.getName() << "' (w/ kernel module: '"
                    << KernelModule.getName() << "')\n");

  // Create a module (KernelModule) to hold all device side functions for all
  // parallel constructs in the module we are processing (M). At present a loop
  // processor will be created for each construct and is then responsible for
  // the steps required to prepare the "kernel" module (KernelModule) for code
  // generation to PTX.

  // Build the details required to have a PTX code generation path ready to go
  // at completion of the module processing; see postProcessModule() for when
  // that stage is kicked off via the Tapir layer.
  std::string ArchString = "nvptx64";
  Triple TT(ArchString, "nvidia", "cuda");

  std::string PTXVersionStr = PTXVersionFromCudaVersion();
  std::string Error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, Error);
  if (!PTXTarget) {
    errs() << "Target lookup failed: " << Error << "\n";
    report_fatal_error("Unable to find registered PTX target. "
                       "Was LLVM built with the NVPTX target enabled?");
  }

  PTXTargetMachine = PTXTarget->createTargetMachine(
      TT.getTriple(), GPUArch, PTXVersionStr.c_str(), TargetOptions(),
      Reloc::PIC_, CodeModel::Large, CodeGenOptLevel::Aggressive);

  KernelModule.setTargetTriple(TT.str());
  KernelModule.setDataLayout(PTXTargetMachine->createDataLayout());

  LLVM_DEBUG(dbgs() << "\ttarget triple: " << TT.getTriple() << "\n");
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

void CudaABI::pushPTXFilename(const std::string &FN) {
  ModulePTXFileList.push_back(FN);
}

void CudaABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}

std::unique_ptr<Module> &CudaABI::getLibDeviceModule() {
  if (not LibDeviceModule) {
    LLVMContext &Ctx = KernelModule.getContext();
    llvm::SMDiagnostic SMD;
    llvm::errs() << "libdevice: " << KITSUNE_CUDA_LIBDEVICE_BC << "\n";
    // KITSUNE FIXME: It might be useful during development to override the
    // libdevice.10.bc function. We could do this with a command-line argument
    // that gets passed to this transform.
    // std::optional<std::string> CudaPath = sys::Process::FindInEnvPath(
    //     "CUDA_HOME", "nvvm/libdevice/libdevice.10.bc");
    // if (!CudaPath) {
    //   CudaPath = sys::Process::FindInEnvPath("CUDA_PATH",
    //                                          "nvvm/libdevice/libdevice.10.bc");
    //   if (!CudaPath)
    //     report_fatal_error("Unable to load cuda libdevice.10.bc!");
    // }

    llvm::StringRef LibDeviceBCFile = KITSUNE_CUDA_LIBDEVICE_BC;
    LibDeviceModule = parseIRFile(LibDeviceBCFile, SMD, Ctx);
    if (not LibDeviceModule)
      report_fatal_error(llvm::StringRef("Failed to parse: ") +
                         LibDeviceBCFile);
  }

  return LibDeviceModule;
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grainsize at 1 has almost always shown to yield the
  // best results in terms of performance.  We have yet to really do
  // a detailed study of the aspects here so consider anything done
  // here as a lot of remaining work and exploration.
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize.getValue());
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // no-op...
  // The CUDA transformations split the code into device and host
  // side modules.
}

void CudaABI::addHelperAttributes(Function &F) { /* no-op */ }

bool CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  return false;
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops) {
    // LLVM_DEBUG(dbgs() << "cuabi: post-processing function '" << F.getName()
    //                   << "'\n");
    // LLVMContext &Ctx = M.getContext();
    // Type *VoidTy = Type::getVoidTy(Ctx);
    // FunctionCallee KitCudaSyncFn =
    //     M.getOrInsertFunction("__kitcuda_sync_current_stream", VoidTy);
    // for (Value *SR : SyncRegList) {
    //   for (Use &U : SR->uses()) {
    //     if (auto *SyncI = dyn_cast<SyncInst>(U.getUser())) {
    //       CallInst::Create(KitCudaSyncFn, {}, "",
    //                        &*SyncI->getSuccessor(0)->begin());
    //       LLVM_DEBUG(dbgs() << "  cuabi: inserting sync call for sync region:
    //       "
    //                         << *SyncI << "\n");
    //     }
    //   }
    // }
    SyncRegList.clear();
  }
}

void CudaABI::postProcessHelper(Function &F) { /* no-op */ }

void CudaABI::preProcessOutlinedTask(llvm::Function &, llvm::Instruction *,
                                     llvm::Instruction *, bool, BasicBlock *) {
  /* no-op */
}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::postProcessRootSpawner(Function &F, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  /* no-op */
}

void CudaABI::preProcessRootSpawner(llvm::Function &, BasicBlock *TFEntry) {
  /* no-op */
}

CudaABIOutputFile CudaABI::assemblePTXFile(CudaABIOutputFile &PTXFile) {

  LLVM_DEBUG(dbgs() << "\t- assembling PTX file '" << PTXFile->getFilename()
                    << "'.\n");

  std::error_code EC;
  llvm::StringRef PTXASExe = KITSUNE_CUDA_PTXAS;
  SmallString<255> AsmFileName(PTXFile->getFilename());
  sys::path::replace_extension(AsmFileName, ".s");
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                             sys::fs::OpenFlags::OF_None);
  pushPTXFilename(PTXFile->getFilename().str());

  // Build the command line for ptxas...  There are some target specific options
  // that we support to configure some specifics here.  See the 'opt' entries
  // near the top of this file.
  // These can be passed to the transform via '-mllvm <cuabi-option>'.
  opt::ArgStringList PTXASArgList;
  PTXASArgList.push_back(PTXASExe.data());

  // TODO: Do we need/want to add support for generating relocatable code?

  // --gpu-name <gpu name>: Specify name of GPU to generate code for.
  // (e.g., 'sm_70','sm_72','sm_75','sm_80','sm_86', 'sm_87')
  PTXASArgList.push_back("--gpu-name"); // target gpu architecture.
  PTXASArgList.push_back(GPUArch.c_str());

  // For now let's always warn if we spill registers...
  PTXASArgList.push_back("--warn-on-spills");
  PTXASArgList.push_back("--verbose");

  if (OptLevel > 3) {
    errs() << "cuabi: warning -- unknown optimization level.  Using level-3.\n";
    OptLevel = 3;
  }

  PTXASArgList.push_back("--opt-level");
  switch (OptLevel) {
  case 0:
    PTXASArgList.push_back("0");
    break;
  case 1:
    PTXASArgList.push_back("1");
    break;
  case 2:
    PTXASArgList.push_back("2");
    break;
  case 3:
    PTXASArgList.push_back("3");
    // PTXASArgList.push_back("--extensible-whole-program");
    break;
  default:
    llvm_unreachable_internal("unhandled/unexpected optimization level",
                              __FILE__, __LINE__);
    break;
  }

  PTXASArgList.push_back("--output-file");
  std::string SCodeFilename = AsmFile->getFilename().str();
  PTXASArgList.push_back(SCodeFilename.c_str());
  std::string ptxfile(PTXFile->getFilename().str());
  PTXASArgList.push_back(ptxfile.c_str());

  // Build argv for exec'ing ptxas...
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);

  auto PTXASArgs = toStringRefArray(PTXASArgv.data());
  LLVM_DEBUG(dbgs() << "\t- ptxas command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  // Finally we are ready to run ptxas...
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(PTXASExe, PTXASArgs, std::nullopt, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("fatal error: 'ptxas' execution failed!");

  if (ExecStat != 0)
    // 'ptxas' ran but returned an error state.
    report_fatal_error("fatal error: 'ptxas' failure: " + StringRef(ErrMsg));

  // TODO: Not sure we need to force 'keep' here as we return
  // the output file but will keep it here for now just to play it
  // safe.
  AsmFile->keep();
  return AsmFile;
}

// We can't create a correct launch sequence until all the kernels within a
// (LLVM) module are generated.  When post-processing the module we create the
// fat binary and then to revisit the kernel launch calls we created at the loop
// level and replace the fat binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the host to the
// device prior to kernel launches.  This requires digging some additonal
// details out of the fat binary (CUDA module).
void CudaABI::finalizeLaunchCalls(Module &M, GlobalVariable *Fatbin) {

  LLVM_DEBUG(dbgs() << "\t- finalizing kernel launch calls...\n");

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  // TODO: Move these callees to the constructor -- no need
  // to build and destory each time... Perhaps speed us up a
  // tidbit...
  FunctionCallee KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // device pointer
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name

  FunctionCallee KitCudaMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_sym_to_device",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            Int64Ty,   // device ptr
                            Int64Ty);  // num bytes

  // Search for kernel launch calls that we built prior to the creation
  // of the fat binary -- which we now have.  Replace the first parameter
  // in each call (which is currently null) with the fat binary pointer.
  LLVM_DEBUG(dbgs() << "\t\tsearching...\n");
  auto &FnList = M.getFunctionList();

  // If we have encountered any attributed launches (forall loops) we
  // will have a call to a dummy runtime entry point
  // (_kitrt_dummy_threads_per_blk) that we use to keep the
  // threads-per-block expression from getting DCE'ed.  When lowered
  // from clang, this call was inserted and we need to use its sole
  // parameter in place of the threads-per-block parameter in the
  // kernel launch call.  These dummy calls should be paired with
  // launch calls, so as we look for a launch we *should* first
  // encounter the threads-per-block call to pair with it.  We keep
  // track of the dummy calls so they can removed at the end of the
  // launch finalization.
  CallInst *ThreadsPerBlockCI = nullptr;
  std::list<CallInst *> DummyCIList;
  CallInst *SavedLaunchCI = nullptr;

  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {

            if (CFn->getName().starts_with("__kitrt_dummy_threads_per_blk")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* discovered a threads-per-block "
                                   "placeholder call.\n");
              assert(ThreadsPerBlockCI == nullptr && "expected null pointer!");
              ThreadsPerBlockCI = CI;
            } else if (CFn->getName().starts_with(
                           "__kitcuda_sync_thread_stream")) {
              LLVM_DEBUG(dbgs()
                         << "\t\t\t* patching sync call: " << *CI << "\n");
              if (SavedLaunchCI != nullptr) {
                AllocaInst *StreamAI = getLaunchStream(SavedLaunchCI);
                assert(StreamAI != nullptr && "unexpected null launch stream!");
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t* stream alloca: " << *StreamAI << "\n");
                IRBuilder<> SyncBuilder(CI);
                Value *CudaStream =
                    SyncBuilder.CreateLoad(VoidPtrTy, StreamAI, "custreamh");
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t* cuda stream: " << *CudaStream << "\n");
                CI->setArgOperand(0, CudaStream);
                if (ReducersLaunchInfo *LaunchInfo =
                        getReducerLaunchInfo(SavedLaunchCI)) {
                  SyncBuilder.SetInsertPoint(CI->getNextNonDebugInstruction());
                  FunctionCallee KitCudaMemCopyAndFreeFromDevFn =
                      M.getOrInsertFunction(
                          "__kitcuda_mem_copy_and_free_from_device",
                          VoidPtrTy,  // return the stream
                          VoidPtrTy,  // device pointer
                          Int64Ty,    // number of bytes to copy
                          VoidPtrTy,  // host pointer
                          VoidPtrTy); // opaque stream
                  Value *DeviceViewBasePtr = SyncBuilder.CreateLoad(
                      VoidPtrTy, LaunchInfo->DeviceViewBasePtrPtr,
                      "device_view_base");
                  SyncBuilder.CreateCall(
                      KitCudaMemCopyAndFreeFromDevFn,
                      {LaunchInfo->HostViewBasePtr,
                       ConstantInt::get(Int64Ty, LaunchInfo->ViewSize),
                       DeviceViewBasePtr, CudaStream});
                  for (auto [Ptr, HostViewPtr, Size, IdFn, MergeFn] :
                       LaunchInfo->ReducerInfos) {
                    // Generate a lookup call to get the current view of the
                    // reducer.
                    Value *HostView = SyncBuilder.CreateIntrinsic(
                        Intrinsic::hyper_lookup, {Int64Ty},
                        {Ptr, ConstantInt::get(Int64Ty, Size), IdFn, MergeFn});
                    SyncBuilder.CreateCall(MergeFn, {HostView, HostViewPtr});
                  }
                }

                // Currently, when __kitcuda_sync_thread_stream is called, the
                // stream is assumed to be done with its work and recycled. So
                // it must not be reused again. This could happen if the
                // launch/sync block is called multiple times in a row should
                // StreamAI retains its current value. Then multiple pointers to
                // the same stream will be added to kitrt's internal queue and
                // cause a SEGFAULT when we attempt to destroy them all due to
                // multiple frees.
                // Store null to StreamAI to prevent this. The stream itself can
                // still be reused internally, but this time after popping from
                // the queue.
                SyncBuilder.CreateStore(ConstantPointerNull::get(VoidPtrTy),
                                        StreamAI);

                SavedLaunchCI = nullptr;
                LLVM_DEBUG(dbgs() << "\t\t\t* patched call: " << *CI << "\n");
              } else {
                LLVM_DEBUG(dbgs()
                           << "\t\t\t- note, matching launch not found!\n");
              }
            } else if (CFn->getName().starts_with("__kitcuda_launch_kernel")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* patching launch: " << *CI << "\n");
              SavedLaunchCI = CI;
              Value *CFatbin;
              CFatbin = CastInst::CreateBitOrPointerCast(Fatbin, VoidPtrTy,
                                                         "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);

              // Only replace TPB arg for regular (non-nested) launches.
              // Nested launches don't have a TPB parameter.
              bool IsNestedLaunch = CFn->getName() == "__kitcuda_launch_kernel_nested";
              if (ThreadsPerBlockCI && !IsNestedLaunch) {
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t replacing thread-per-block arg.\n");
                Value *TBPParam = ThreadsPerBlockCI->getArgOperand(0);
                assert(TBPParam && "unexpected null arg operand!");
                CI->setArgOperand(4, TBPParam);
                DummyCIList.push_back(ThreadsPerBlockCI);
                ThreadsPerBlockCI = nullptr;
              }

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              assert(NI && "unexpected null instruction!");

              // We need to explicitly add code to sync up host- and
              // device-side global values prior to launching kernels.
              // We only have a complete awareness of this now so insert
              // the supporting runtime calls.
              //
              // TODO: This is overdone -- we copy *all* globals and not
              // just those that the kernel we're launching is using.
              //
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + "_devvar";
                LLVM_DEBUG(dbgs() << "\t\t\t  processing global: '"
                                  << HostGV->getName() << "'\n");
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitCudaGetGlobalSymbolFn,
                                     {CFatbin, SymName}, ".cuabi_devptr", NI);
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitCudaMemcpyToDeviceFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    NI);
              }
            } else if (CFn->getName().starts_with("__kitcuda_alloc_scanner")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* patching __kitcuda_alloc_scanner: "
                                << *CI << "\n");
              SavedLaunchCI = CI;
              Value *CFatbin = CastInst::CreateBitOrPointerCast(
                  Fatbin, VoidPtrTy, "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);
            }
          }
        }
      }
    }
  }

  for (auto I : DummyCIList) {
    LLVM_DEBUG(dbgs() << "\t\t\t erasing dummy threads-per-block call.\n");
    I->eraseFromParent();
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable("_cuabi.dummy_fatbin", true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(Fatbin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\tcleaning up dummy fatbin global.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    // FIXME: If we haven't found the proxy for a fat binary the odds are we
    // have not found any parallel loops in the code...  Technically, this
    // should not be seen as a compiler error...
    report_fatal_error("internal error! unable to find proxy fatbin pointer!");
  }
}

CudaABIOutputFile CudaABI::createFatbinaryFile(CudaABIOutputFile &AsmFile) {
  std::error_code EC;
  SmallString<255> FatbinFilename(AsmFile->getFilename());
  sys::path::replace_extension(FatbinFilename, ".cufatbin");
  CudaABIOutputFile FatbinFile;
  FatbinFile = std::make_unique<ToolOutputFile>(FatbinFilename, EC,
                                                sys::fs::OpenFlags::OF_None);

  LLVM_DEBUG(dbgs() << "\t- generatng fatbinary image file '"
                    << FatbinFile->getFilename() << "'.\n");

  // // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  // auto FatbinaryExe = sys::findProgramByName("fatbinary");
  // if ((EC = FatbinaryExe.getError()))
  //   report_fatal_error("'fatbinary' not found. "
  //                      "Is a CUDA installation in your path?");

  llvm::StringRef FatbinaryExe = KITSUNE_CUDA_FATBINARY;
  opt::ArgStringList FatbinaryArgList;
  FatbinaryArgList.push_back(FatbinaryExe.data());
  FatbinaryArgList.push_back("--64");
  FatbinaryArgList.push_back("--create");
  FatbinaryArgList.push_back(FatbinFilename.c_str());

  std::string FatbinaryImgArgs =
      "--image=profile=" + GPUArch + ",file=" + AsmFile->getFilename().str();
  FatbinaryArgList.push_back(FatbinaryImgArgs.c_str());

  std::list<std::string> PTXFilesArgList;
  if (EmbedPTXInFatbinaries) {
    std::string VArchStr = virtualArchForCudaArch(GPUArch);
    if (VArchStr == "unknown")
      report_fatal_error("cuabi: no virtual target for given gpuarch '" +
                         StringRef(GPUArch) + "'!");

    std::string PTXFixedArgStr = "--image=profile=" + VArchStr + ",file=";
    for (auto &PTXFile : ModulePTXFileList) {
      std::string arg = PTXFixedArgStr + PTXFile;
      std::list<std::string>::const_iterator it;
      it = PTXFilesArgList.emplace(PTXFilesArgList.end(), std::move(arg));
      FatbinaryArgList.push_back(it->c_str());
    }
  }

  FatbinaryArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatbinaryArgv;
  FatbinaryArgv.append(FatbinaryArgList.begin(), FatbinaryArgList.end());
  auto FatbinaryArgs = toStringRefArray(FatbinaryArgv.data());

  LLVM_DEBUG(dbgs() << "\tfatbinary command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : FatbinaryArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat =
      sys::ExecuteAndWait(FatbinaryExe, FatbinaryArgs, std::nullopt, {},
                          0, /* secs to wait -- 0 --> unlimited */
                          0, /* memory limit -- 0 --> unlimited */
                          &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (ExecStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary' returns to the
    // environment -- currently assuming it matches standard practices...
    report_fatal_error("'fatbinary' error:" + StringRef(ErrMsg));

  if (EmbedPTXInFatbinaries) {
    std::list<std::string>::iterator it = PTXFilesArgList.begin();
    while (it != PTXFilesArgList.end()) {
      PTXFilesArgList.erase(it++);
    }
  }

  // TODO: Not sure we need to force 'keep' here as we return the output file
  // but will keep it here for now just to play it safe.
  FatbinFile->keep();
  return FatbinFile;
}

GlobalVariable *CudaABI::embedFatbinary(CudaABIOutputFile &FatbinaryFile) {

  LLVM_DEBUG(dbgs() << "\t- code gen for embedded fat binary image...\n");

  // Allocate a buffer to store the fat binary image in.  We
  // will then codegen it into the host-side module.
  std::unique_ptr<llvm::MemoryBuffer> Fatbinary = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FBBufferOrErr =
      MemoryBuffer::getFile(FatbinaryFile->getFilename());
  if (std::error_code EC = FBBufferOrErr.getError()) {
    report_fatal_error("cuabi: failed to load fat binary image: " +
                       StringRef(EC.message()));
  }
  Fatbinary = std::move(FBBufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\t\treading fat binary.  size = "
                    << Fatbinary->getBufferSize() << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FatbinArray = ConstantDataArray::getRaw(
      StringRef(Fatbinary->getBufferStart(), Fatbinary->getBufferSize()),
      Fatbinary->getBufferSize(), Int8Ty);

  LLVM_DEBUG(dbgs() << "\t\tcreating associated global 'fatbin' variable.\n");

  // Create a global variable to hold the fatbinary image.
  GlobalVariable *FatbinaryGV;
  FatbinaryGV = new GlobalVariable(M, FatbinArray->getType(), true,
                                   GlobalValue::PrivateLinkage, FatbinArray,
                                   "_cuabi_fatbin_ptr");
  return FatbinaryGV;
}

void CudaABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = Int64Ty;
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  FunctionCallee RegisterVarFn = M.getOrInsertFunction(
      "__cudaRegisterVar", VoidTy, VoidPtrPtrTy, CharPtrTy, CharPtrTy,
      CharPtrTy, IntTy, VarSizeTy, IntTy, IntTy);
  for (auto &HostGV : GlobalVars) {
    uint64_t VarSize = DL.getTypeAllocSize(HostGV->getType());
    Value *VarName = tapir::createConstantStr(HostGV->getName().str(), M);
    std::string DevVarName = HostGV->getName().str() + "_devvar";
    Value *DevName = tapir::createConstantStr(DevVarName, M, DevVarName);
    llvm::Value *Args[] = {
        Handle,
        B.CreateBitCast(HostGV, VoidPtrTy),
        VarName,
        DevName,
        ConstantInt::get(IntTy, 0), // HostGV->isExternalLinkage()),
        ConstantInt::get(VarSizeTy, VarSize),
        ConstantInt::get(IntTy, HostGV->isConstant()),
        ConstantInt::get(IntTy, 0)};

    LLVM_DEBUG(dbgs() << "\t\t\thost global '" << HostGV->getName().str()
                      << "' to device '" << DevVarName << "'.\n");
    B.CreateCall(RegisterVarFn, Args);
  }
}

Function *CudaABI::createCtor(GlobalVariable *Fatbinary,
                              GlobalVariable *Wrapper) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *BoolTy = Type::getInt8Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".ctor." + KernelModule.getName(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  FunctionCallee KitRTSetDefaultMaxTheadsPerBlockFn = M.getOrInsertFunction(
      "__kitcuda_set_default_threads_per_blk", VoidTy, IntTy);
  CtorBuilder.CreateCall(KitRTSetDefaultMaxTheadsPerBlockFn,
                         {ConstantInt::get(IntTy, MaxThreadsPerBlock)});

  FunctionCallee KitCudaInitFn =
      M.getOrInsertFunction("__kitcuda_initialize", VoidTy);
  CtorBuilder.CreateCall(KitCudaInitFn, {});

  FunctionCallee KitCudaOccLaunchFn =
      M.getOrInsertFunction("__kitcuda_use_occupancy_launch", VoidTy, BoolTy);
  Value *EnableOccLaunches;
  if (UseOccupancyLaunches)
    EnableOccLaunches = ConstantInt::get(BoolTy, 1);
  else
    EnableOccLaunches = ConstantInt::get(BoolTy, 0);
  CtorBuilder.CreateCall(KitCudaOccLaunchFn, {EnableOccLaunches});

  // TODO: The parameters to the CUDA registration calls can be opaque about
  // specifics (e.g., types).  Once we sort out some details we should clean
  // this up.

  // The general layout of the calls for fat binary registration
  // look something like this:
  //
  // void** __cudaRegisterFatBinary(void *fatCubin);
  //
  // void __cudaRegisterVar(void **fatCubinHandle,
  //                        char  *hostVar,
  //                        char  *deviceAddress,
  //                        const char  *deviceName,
  //                        int    ext,
  //                        size_t size,
  //                        int    constant,
  //                        int    global);
  //
  // void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
  //
  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__cudaRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, // cubin handle.
                                              VoidPtrTy, // fat bin device txt.
                                              false));
  CallInst *RegFatbin = CtorBuilder.CreateCall(
      RegisterFatbinaryFn, CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));

  GlobalVariable *Handle = new GlobalVariable(
      M, VoidPtrPtrTy, false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), CUABI_PREFIX + ".fbhand");
  Handle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerABIAlignment(0));
  Handle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  Value *HandlePtr = CtorBuilder.CreateLoad(VoidPtrPtrTy, Handle,
                                            CUABI_PREFIX + ".fbhand.ptr");

  // TODO: It is not 100% clear what calls we actually need to make here for
  // kernel, variable, etc. registration with CUDA.  Clang makes these calls but
  // we are targeting CUDA driver API entry points via the Kitsune runtime
  // library so these calls are potentially unneeded...
  if (!GlobalVars.empty()) {
    LLVM_DEBUG(dbgs() << "\t\tbinding host and device global variables...\n");
    bindGlobalVariables(HandlePtr, CtorBuilder);
  }

  // Wrap up fatbinary registration steps...
  FunctionCallee EndFBRegistrationFn =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd",
                            FunctionType::get(VoidTy,
                                              VoidPtrPtrTy, // cubin handle.
                                              false));
  CtorBuilder.CreateCall(EndFBRegistrationFn, RegFatbin);

  // Now add a Dtor to help us clean up at program exit...
  if (Function *CleanupFn = createDtor(Handle)) {
    // Hook into 'atexit()'...
    FunctionType *AtExitFnTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFn =
        M.getOrInsertFunction("atexit", AtExitFnTy, AttributeList());
    CtorBuilder.CreateCall(AtExitFn, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return CtorFn;
}

Function *CudaABI::createDtor(GlobalVariable *FBHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__cudaUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".dtor", &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, FBHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kitcuda_destroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});

  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void CudaABI::registerFatbinary(GlobalVariable *Fatbinary) {

  LLVM_DEBUG(dbgs() << "\t- registering fat binary...\n");

  // Registering the fat binary image (and all the associated components) is an
  // undocumented portion of the CUDA API. One place to peek for some details
  // hides in the cuda header files; specifially fatbinary_section.h. This
  // shows the following struct that we need to have in the host side code.
  //
  //    struct fatbinC_Wrapper_t {
  //      int magic;
  //      int version;
  //      const unsigned long long *data;
  //      void *filename_or_fatbins;
  //    };
  //
  // * Per the header, the magic number is 0x466243B1.
  // * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2.
  //   (more below)
  // * Then section and segments are needed that contains
  //   the "fatbin control structure".  This loosely looks
  //   like:
  //
  //        Control section name: ".nvFatBinSegment"
  //        Fatbinary section name: ".nv_fatbin"
  //        Pre-linked relocatable section: "__nv_relfatbin"
  //
  // * The last struct member varies between versions.  In the
  //   case of version 1 it can be a offline filename and for
  //   version 2 it is an array of pre-linked fatbins.
  //
  const int FATBINARY_MAGIC_ID = 0x466243B1;
  const int FATBINARY_VERSION = 1;
  const char *FATBIN_CONTROL_SECTION_NAME = ".nvFatBinSegment";
  const char *FATBIN_DATA_SECTION_NAME = ".nv_fatbin";

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  Type *FatbinStrTy = Fatbinary->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(FatbinStrTy), 0),
                       ConstantInt::get(DL.getIndexType(FatbinStrTy), 0)};

  Fatbinary->setSection(FATBIN_DATA_SECTION_NAME);
  Constant *FatbinaryPtr = ConstantExpr::getGetElementPtr(
      Fatbinary->getValueType(), Fatbinary, Zeros);

  // Wrap the fatbinary in struct that the CUDA runtime and tools expect
  // to exist in final objects/executables.
  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // data
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS = ConstantStruct::get(
      WrapperTy, ConstantInt::get(IntTy, FATBINARY_MAGIC_ID),
      ConstantInt::get(IntTy, FATBINARY_VERSION), FatbinaryPtr,
      ConstantPointerNull::get(VoidPtrTy));

  GlobalVariable *Wrapper =
      new GlobalVariable(M, WrapperTy, true, GlobalValue::InternalLinkage,
                         WrapperS, "_cuabi_wrapper");
  Wrapper->setSection(FATBIN_CONTROL_SECTION_NAME);
  Wrapper->setAlignment(Align(DL.getPrefTypeAlign(Wrapper->getType())));

  // The rest of the registration details are tucked into a constructor
  // entry...
  LLVM_DEBUG(dbgs() << "\t\tcode gen new global ctor entry...\n");
  Function *CtorFn = createCtor(Fatbinary, Wrapper);
  if (CtorFn) {
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M, ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  }
}

namespace {

// NVPTX backend does not lower LLVM fence instructions properly.  See
// https://github.com/llvm/llvm-project/issues/61411 We fix it by replacing
// fence instructions with membar.gl (__threadfence).  This is always correct,
// because __threadfence has seqcst semantics, not weaker than any fence.
void fixFencesAndAtomics(Module &KernelModule) {
  for (Function &F : KernelModule) {
    SmallVector<FenceInst *> Fences;
    for (Instruction &I : instructions(F)) {
      if (FenceInst *FI = dyn_cast<FenceInst>(&I)) {
        Fences.push_back(FI);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
        if (SI->isAtomic()) {
          // Mark it as not atomic.
          // TODO: maybe make it into an atomicCAS is more appropriate?
          SI->setAtomic(AtomicOrdering::NotAtomic);
        }
      }
    }
    for (FenceInst *FI : Fences) {
      IRBuilder<> Builder(FI);
      Value *MemBar =
          Builder.CreateIntrinsic(Intrinsic::nvvm_membar_gl, {}, {});
      LLVM_DEBUG(dbgs() << "\t\t\t* replacing fence with membar.gl: " << *FI
                        << "\n");
      FI->replaceAllUsesWith(MemBar);
      // TODO: copy over debug info?
      FI->eraseFromParent();
    }
  }
}

} // namespace

// --- Kitsune user shared-memory support ------------------------------------
// Lower calls to the marker function `void *__kitcuda_block_shared(i64 nbytes)`
// (declared extern in user code) into per-block CUDA *static* shared memory.
// Each call becomes an addrspace(3) global (identical mechanism to the reducer
// block arrays) whose generic-cast pointer replaces the call, so ordinary
// loads/stores through it lower to ld.shared/st.shared. Because this is static
// shared memory, the launch's dynamic-shared-bytes argument stays 0 and the
// occupancy calculator (cuOccupancyMaxPotentialBlockSize) automatically folds
// the per-block usage into the chosen block size. `nbytes` must be a
// compile-time constant.
static void lowerBlockSharedCalls(Module &KM) {
  Function *F = KM.getFunction("__kitcuda_block_shared");
  if (!F)
    return;
  LLVMContext &Ctx = KM.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  SmallVector<CallInst *, 8> Calls;
  for (User *U : F->users())
    if (auto *CI = dyn_cast<CallInst>(U))
      Calls.push_back(CI);
  unsigned Idx = 0;
  for (CallInst *CI : Calls) {
    auto *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    if (!SizeC) {
      Ctx.emitError("kitsune: __kitcuda_block_shared size must be a "
                    "compile-time constant");
      continue;
    }
    uint64_t NBytes = SizeC->getZExtValue();
    if (NBytes == 0)
      NBytes = 1;
    ArrayType *ArrTy = ArrayType::get(Int8Ty, NBytes);
    GlobalVariable *GV = new GlobalVariable(
        KM, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(ArrTy), "__kitcuda_smem." + Twine(Idx++), nullptr,
        GlobalVariable::NotThreadLocal, /*AddressSpace=*/3);
    GV->setAlignment(Align(16));
    IRBuilder<> B(CI);
    Value *P3 = B.CreateInBoundsGEP(ArrTy, GV, {B.getInt32(0), B.getInt32(0)});
    Value *Gen = B.CreateAddrSpaceCast(P3, CI->getType());
    CI->replaceAllUsesWith(Gen);
    CI->eraseFromParent();
  }
  LLVM_DEBUG(dbgs() << "cuabi: lowered " << Idx
                    << " __kitcuda_block_shared call(s) to shared memory\n");
}

CudaABIOutputFile CudaABI::generatePTX() {
  // Lower Kitsune user shared-memory requests to static addrspace(3) globals
  // before PTX generation.
  lowerBlockSharedCalls(KernelModule);

  // Link in the Bitcode file
  if (UseKitCudaRuntimeBC) {
    SMDiagnostic SMD;
    LLVMContext &C = KernelModule.getContext();
    // Parse the bitcode file.  This call imports structure definitions, but not
    // function definitions.
    if (std::unique_ptr<Module> ExternalModule =
            parseIRFile(ClKitCudaRuntimeBCPath, SMD, C)) {
      // Link the external module into the current module, copying over global
      // values.
      bool Fail = Linker::linkModules(KernelModule, std::move(ExternalModule),
                                      Linker::Flags::LinkOnlyNeeded);
      if (Fail)
        C.emitError("CudaABI: Failed to link bitcode ABI file: " +
                    Twine(ClKitCudaRuntimeBCPath));
    } else {
      C.emitError("CudaABI: Failed to parse bitcode ABI file: " +
                  Twine(ClKitCudaRuntimeBCPath));
    }
  }

  LLVM_DEBUG(dbgs() << "\t- generating PTX...\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                 ".post.preopt.ll"));

  // Take the intermediate form code in the kernel module and
  // generate a PTX file.  The PTX file will be named the same as
  // the original input source module (M) with the extension changed
  // to PTX.
  std::string ModelPTXFileName =
      std::string(CUABI_PREFIX) + "%%-%%-%%_" + KernelModule.getName().str();
  SmallString<1024> PTXFileName;
  sys::fs::createUniquePath(ModelPTXFileName.c_str(), PTXFileName, true);
  sys::path::replace_extension(PTXFileName, ".ptx");

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> PTXFile;
  PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                             sys::fs::OpenFlags::OF_None);
  PTXFile->keep();

  KernelModule.setModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", true);
  if (OptLevel > 0) {
    if (OptLevel > 3)
      OptLevel = 3;
    LLVM_DEBUG(dbgs() << "\t- running kernel module optimization passes...\n");
    PipelineTuningOptions PTO;
    PTO.LoopVectorization = OptLevel > 2;
    PTO.SLPVectorization = OptLevel > 2;
    PTO.LoopUnrolling = OptLevel > 2;
    PTO.LoopInterleaving = OptLevel > 2;
    PTO.LoopStripmine = OptLevel > 2;
    OptimizationLevel OptLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };

    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassInstrumentationCallbacks *PIC = new PassInstrumentationCallbacks();
    PassBuilder PB(PTXTargetMachine, PTO, std::nullopt, PIC);

    // Add instrumentation to print IR after each pass

    LLVM_DEBUG({
      if (verifyModule(KernelModule, &errs())) {
        report_fatal_error("Module is broken before optimization passes");
      }
      std::string DumpDir = "ir-dumps-" + KernelModule.getName().str();
      sys::fs::create_directory(DumpDir);
      int PassCount = 0;
      const auto DumpIRAfterPass = [&](StringRef PassID, Any IR,
                                       const PreservedAnalyses &PA) {
        if (const auto *MPtr = any_cast<const Module *>(&IR)) {
          const auto *M = *MPtr;
          std::string FileName = DumpDir + "/" + (PassCount < 10 ? "0" : "") +
                                 std::to_string(PassCount) + "-after-" +
                                 PassID.str() + ".ll";
          PassCount++;
          std::error_code EC;
          raw_fd_ostream OS(FileName, EC, sys::fs::OF_None);
          if (!EC) {
            M->print(OS, nullptr);
          }
          if (verifyModule(*M, &errs())) {
            report_fatal_error(Twine{"Module is broken after pass: "} +
                               PassID.str());
          }
        }
      };
      PIC->registerAfterPassCallback(DumpIRAfterPass);
    });

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PTXTargetMachine->registerPassBuilderCallbacks(PB);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM =
        PB.buildPerModuleDefaultPipeline(OptLevels[OptLevel]);
    MPM.addPass(VerifierPass());
    // MPM.printPipeline(dbgs(), [](StringRef Name) { return Name; });
    LLVM_DEBUG(dbgs() << "\t\t* module: " << KernelModule.getName() << "\n");
    MPM.run(KernelModule, MAM);
    LLVM_DEBUG(dbgs() << "\t\tpasses complete.\n");
    LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                   ".postopt.LTO.ll"));
  }

  // Fixup thread fence instructions.
  fixFencesAndAtomics(KernelModule);

  // Mark loads from readonly pointer arguments with !invariant.load metadata.
  // The NVPTX backend converts these to ld.global.nc (read-only cache) which
  // bypasses L1 and uses the dedicated texture cache, improving throughput for
  // read-only data like twiddle tables.
  {
    MDNode *InvariantMD = MDNode::get(KernelModule.getContext(), {});
    for (Function &F : KernelModule) {
      if (F.isDeclaration())
        continue;
      // Check nvvm.annotations to confirm this is a kernel
      bool IsKernel = false;
      if (auto *NMD = KernelModule.getNamedMetadata("nvvm.annotations")) {
        for (auto *Op : NMD->operands()) {
          if (Op->getNumOperands() >= 3)
            if (auto *VMD = dyn_cast<ValueAsMetadata>(Op->getOperand(0).get()))
              if (VMD->getValue() == &F)
                IsKernel = true;
        }
      }
      if (!IsKernel)
        continue;
      for (unsigned I = 0, E = F.arg_size(); I < E; ++I) {
        Argument *Arg = F.getArg(I);
        if (!Arg->getType()->isPointerTy())
          continue;
        // Check if this pointer arg is only used for loads (never stored to).
        // If so, mark it readonly+noalias here (right before codegen) so the
        // NVPTX backend will emit ld.global.nc for it.
        bool OnlyLoads = true;
        bool HasLoad = false;
        SmallVector<Value *, 16> PtrWorklist;
        SmallPtrSet<Value *, 16> PtrVisited;
        PtrWorklist.push_back(Arg);
        while (!PtrWorklist.empty() && OnlyLoads) {
          Value *V = PtrWorklist.pop_back_val();
          if (!PtrVisited.insert(V).second) continue;
          for (User *UU : V->users()) {
            if (isa<LoadInst>(UU)) {
              HasLoad = true;
            } else if (auto *SI = dyn_cast<StoreInst>(UU)) {
              if (SI->getPointerOperand() == V) { OnlyLoads = false; break; }
            } else if (isa<GetElementPtrInst>(UU) || isa<BitCastInst>(UU) ||
                       isa<AddrSpaceCastInst>(UU)) {
              PtrWorklist.push_back(UU);
            } else {
              OnlyLoads = false; break;
            }
          }
        }
        if (!OnlyLoads || !HasLoad)
          continue;
        F.addParamAttr(I, Attribute::ReadOnly);
        F.addParamAttr(I, Attribute::NoAlias);
        // Walk through GEPs and annotate loads as invariant
        SmallVector<Value *, 16> Worklist;
        SmallPtrSet<Value *, 16> Visited;
        Worklist.push_back(Arg);
        while (!Worklist.empty()) {
          Value *V = Worklist.pop_back_val();
          if (!Visited.insert(V).second)
            continue;
          for (User *U : V->users()) {
            if (auto *LI = dyn_cast<LoadInst>(U)) {
              LI->setMetadata(LLVMContext::MD_invariant_load, InvariantMD);
            } else if (isa<GetElementPtrInst>(U) ||
                       isa<BitCastInst>(U) ||
                       isa<AddrSpaceCastInst>(U)) {
              Worklist.push_back(U);
            }
          }
        }
      }
    }
  }

  // PTXAS does not support optimized debug info. Practically, this means
  // DICompileUnit::FullDebug and DICompileUnit::LineTablesOnly do not work. But
  // DICompileUnit::DebugDirectivesOnly does and it's good enough (you get to
  // see the source code from Nsight reports). So we temporarily downgrade the
  // EmissionKind of debug compile units in KernelModule.
  DenseMap<DICompileUnit *, DICompileUnit::DebugEmissionKind> EmissionKindMap;
  for (DICompileUnit *CU : KernelModule.debug_compile_units()) {
    EmissionKindMap[CU] = CU->getEmissionKind();
    switch (CU->getEmissionKind()) {
    case DICompileUnit::NoDebug:
      break;
    case DICompileUnit::DebugDirectivesOnly:
    case DICompileUnit::LineTablesOnly:
    case DICompileUnit::FullDebug:
      CU->setEmissionKind(DICompileUnit::DebugDirectivesOnly);
      break;
    }
  }

  // Hack: strip debug symbols because apparently some NVPTX internal lowering
  // passes handle them incorrectly... Don't have time to fix NVPTX internals.
  StripDebugInfo(KernelModule);

  // Setup the passes and request that the output goes to the
  // specified PTX file.
  LLVM_DEBUG(dbgs() << "\t- PTX file: '" << PTXFileName << "'.\n");
  legacy::PassManager PassMgr;
  if (PTXTargetMachine->addPassesToEmitFile(PassMgr, PTXFile->os(), nullptr,
                                            CodeGenFileType::AssemblyFile,
                                            false))
    report_fatal_error("Cuda ABI transform -- PTX generation failed!");
  PassMgr.run(KernelModule);
  LLVM_DEBUG(dbgs() << "\tkernel optimizations and code gen complete.\n\n");
  LLVM_DEBUG(dbgs() << "\t\tPTX file: " << PTXFile->getFilename() << "\n");

  // Restore the original debug emission kind for debug compile units, as they
  // may still be used by host-side code gen later.
  for (auto &[CU, EmissionKind] : EmissionKindMap) {
    CU->setEmissionKind(EmissionKind);
  }
  return PTXFile;
}

void CudaABI::postProcessModule() {
  // At this point, all tapir constructs in the input module (M) have been
  // transformed (i.e., outlined) into the kernel module. We can now wrap up
  // module-wide changes for both modules and generate a GPU binary.
  // NOTE: postProcessModule() will not be called in cases where parallelism
  // was not discovered during loop spawning.
  LLVM_DEBUG(dbgs() << "\n\n"
                    << "cuabi: postprocessing the kernel '"
                    << KernelModule.getName() << "' and input '" << M.getName()
                    << "' modules.\n");
  { // Remove all personality functions
    for (Function &F : KernelModule) {
      if (F.isDeclaration())
        continue;
      if (F.hasPersonalityFn() && F.getNumOperands()) {
        F.setPersonalityFn(nullptr);
      }
    }
  }

  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                 ".post.unoptimized"));
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".outline-debug"));

  auto L = Linker(KernelModule);
  if (LibDeviceModule) {
    LLVM_DEBUG(dbgs() << "\t- linking in cuda libdevice into kernel module.\n");
    L.linkInModule(std::move(LibDeviceModule), Linker::LinkOnlyNeeded);
  }

  CudaABIOutputFile PTXFile = generatePTX();
  CudaABIOutputFile AsmFile = assemblePTXFile(PTXFile);
  CudaABIOutputFile FatbinFile = createFatbinaryFile(AsmFile);
  GlobalVariable *Fatbinary = embedFatbinary(FatbinFile);

  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".post-fatbin"));

  finalizeLaunchCalls(M, Fatbinary);

  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".post-finalize-launch"));

  registerFatbinary(Fatbinary);
  if (HostOptLevel > 0) {
    if (HostOptLevel > 3)
      HostOptLevel = 3;
    PipelineTuningOptions pto;
    pto.LoopVectorization = HostOptLevel > 2;
    pto.SLPVectorization = HostOptLevel > 2;
    pto.LoopUnrolling = true;
    pto.LoopInterleaving = true;
    pto.LoopStripmine = false;

    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[HostOptLevel];
    PassBuilder pb(PTXTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    PTXTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    mpm.run(M, mam);
    LLVM_DEBUG(dbgs() << "\tpasses complete.\n");
  }

  if (not KeepIntermediateFiles) {
    sys::fs::remove(PTXFile->getFilename());
    sys::fs::remove(AsmFile->getFilename());
    sys::fs::remove(FatbinFile->getFilename());
  }
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  // Create a CUDA loop outline processor for transforming parallel tapir loop
  // constructs into suitable GPU device code.  We hand the outliner the kernel
  // module (KernelModule) as the destination for all generated (device-side)
  // code.

  Loop *TheLoop = TL->getLoop();
  Function *ParentFn = TheLoop->getHeader()->getParent();

  // Skip loops already inside a kernel (they use thread-level parallelism).
  if (ParentFn->getName().starts_with(CUABI_KERNEL_NAME_PREFIX)) {
    LLVM_DEBUG(dbgs() << "cuabi: skipping outline of nested loop at "
                      << TheLoop->getStartLoc() << "\n"
                      << "        (already inside kernel, will use threads)\n");
    return nullptr;
  }

  // Skip loops nested inside another Tapir loop (detected by detach in parent header).
  Loop *ParentLoop = TheLoop->getParentLoop();
  while (ParentLoop) {
    BasicBlock *ParentHeader = ParentLoop->getHeader();
    for (Instruction &I : *ParentHeader) {
      if (isa<DetachInst>(&I)) {
        LLVM_DEBUG(dbgs() << "cuabi: skipping outline of nested tapir loop\n");
        return nullptr;
      }
    }
    ParentLoop = ParentLoop->getParentLoop();
  }

  // Skip thread-parallel inner loops (they stay inside the parent kernel).
  if (isThreadParallelLoop(TheLoop)) {
    LLVM_DEBUG(dbgs() << "cuabi: skipping outline of threadparallel loop at "
                      << TheLoop->getStartLoc() << "\n"
                      << "        (will remain inside parent kernel)\n");
    return nullptr;
  }

  std::string ModuleName = sys::path::filename(M.getName()).str();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KernelName = Fn->getName().str();

  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use a line number
    // based naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    // ModuleName could contain dot which is invalid PTX identifier, so clean it
    // up first.
    KernelName = CUABI_KERNEL_NAME_PREFIX + cleanUpName(ModuleName) + "_" +
                 Twine(LineNumber).str();
  } else {
    // SmallString<255> ModName(Twine(ModuleName).str());
    // sys::path::replace_extension(ModName, "");
    // KernelName = CUABI_PREFIX + ModName.c_str();
    //  In the non-debug mode we use a consecutive numbering scheme for our
    //  kernel names (this is currently handled via the 'make unique'
    //  parameter).
    KernelName = CUABI_KERNEL_NAME_PREFIX + KernelName;
  }

  CudaLoop *Outliner = new CudaLoop(M, KernelModule, KernelName, this);
  return Outliner;
}
