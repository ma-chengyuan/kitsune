#include "kitcuda.h"
#include "kitcuda_dylib.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

struct CUfunc_st {
  unsigned ThreadsPerBlock;
};
struct CUmod_st {};
struct CUstream_st {};

namespace {

int Image;
CUmod_st Module;
CUstream_st Stream;
const char KernelAName[] = "kernel_a";
const char KernelBName[] = "kernel_b";
const char KernelCName[] = "kernel_c";
CUfunc_st KernelA{128};
CUfunc_st KernelB{256};
CUfunc_st KernelC{64};
constexpr CUdeviceptr GlobalSymbol = 4096;
std::atomic<unsigned> ModuleLoads{0};
std::atomic<unsigned> KernelLookups{0};
std::atomic<unsigned> OccupancyQueries{0};

void check(bool Condition, const char *Message) {
  if (!Condition) {
    std::fprintf(stderr, "%s\n", Message);
    std::abort();
  }
}

CUresult loadModule(CUmodule *Result, const void *Data) {
  check(Data == &Image, "unexpected CUDA image");
  *Result = &Module;
  ModuleLoads.fetch_add(1, std::memory_order_relaxed);
  return CUDA_SUCCESS;
}

CUresult getFunction(CUfunction *Result, CUmodule Loaded, const char *Name) {
  check(Loaded == &Module, "unexpected CUDA module");
  if (Name == KernelAName)
    *Result = &KernelA;
  else if (Name == KernelBName)
    *Result = &KernelB;
  else {
    check(Name == KernelCName, "unexpected kernel name");
    *Result = &KernelC;
  }
  KernelLookups.fetch_add(1, std::memory_order_relaxed);
  return CUDA_SUCCESS;
}

CUresult getGlobal(CUdeviceptr *Result, size_t *Size, CUmodule Loaded,
                   const char *) {
  check(Loaded == &Module, "unexpected CUDA module");
  *Result = GlobalSymbol;
  *Size = sizeof(int);
  return CUDA_SUCCESS;
}

CUresult occupancy(int *MinGrid, int *Threads, CUfunction Function,
                   CUoccupancyB2DSize, size_t, int) {
  *MinGrid = 1;
  *Threads = Function->ThreadsPerBlock;
  OccupancyQueries.fetch_add(1, std::memory_order_relaxed);
  return CUDA_SUCCESS;
}

CUresult launch(CUfunction Function, unsigned Blocks, unsigned, unsigned,
                unsigned Threads, unsigned, unsigned, unsigned, CUstream,
                void **Args, void **) {
  uint64_t Trips = *static_cast<uint64_t *>(Args[0]);
  unsigned ExpectedThreads = Function->ThreadsPerBlock;
  check(Threads == ExpectedThreads, "unexpected threads per block");
  check(Blocks == (Trips - 1) / ExpectedThreads + 1,
        "unexpected blocks per grid");
  return CUDA_SUCCESS;
}

void launchKernel(const char *Name, uint64_t Trips) {
  void *Args[] = {&Trips};
  check(__kitcuda_launch_kernel(&Image, Name, Args, Trips, 0, nullptr,
                                &Stream) == &Stream,
        "unexpected CUDA stream");
}

} // namespace

decltype(cuModuleLoadData_p) cuModuleLoadData_p = loadModule;
decltype(cuModuleGetFunction_p) cuModuleGetFunction_p = getFunction;
decltype(cuModuleGetGlobal_v2_p) cuModuleGetGlobal_v2_p = getGlobal;
decltype(cuOccupancyMaxPotentialBlockSize_p)
    cuOccupancyMaxPotentialBlockSize_p = occupancy;
decltype(cuLaunchKernel_p) cuLaunchKernel_p = launch;
decltype(cuDeviceGetAttribute_p) cuDeviceGetAttribute_p = nullptr;
decltype(cuGetErrorName_p) cuGetErrorName_p = nullptr;
decltype(cuGetErrorString_p) cuGetErrorString_p = nullptr;
// Scanner functions are linked with the launch code but unused by this test.
decltype(cuMemAllocAsync_p) cuMemAllocAsync_p = nullptr;
decltype(cuMemFreeAsync_p) cuMemFreeAsync_p = nullptr;
decltype(cuMemsetD8Async_p) cuMemsetD8Async_p = nullptr;

extern "C" {
int _kitcuda_device_id = 0;
bool _kitrt_verbose_mode = false;
void __kitcuda_set_context() {}
void *__kitcuda_get_thread_stream() { return &Stream; }
void __kitrt_print_stack_trace() { std::abort(); }
}

int main() {
  __kitcuda_refine_occupancy_launches(false);

  // Resolve both kernels and prime the launch cache before starting workers.
  constexpr uint64_t HotTrips = 300;
  launchKernel(KernelAName, HotTrips);
  launchKernel(KernelBName, HotTrips);

  constexpr unsigned Workers = 4;
  constexpr unsigned TripsPerWorker = 128;
  std::array<std::thread, Workers> Threads;
  std::atomic<unsigned> Ready{0};
  std::atomic<bool> Start{false};
  for (unsigned Worker = 0; Worker < Workers; ++Worker) {
    Threads[Worker] = std::thread([&, Worker] {
      Ready.fetch_add(1, std::memory_order_release);
      while (!Start.load(std::memory_order_acquire))
        std::this_thread::yield();

      for (unsigned I = 0; I < TripsPerWorker; ++I) {
        uint64_t ColdTrips = HotTrips + 1 + Worker * TripsPerWorker + I;
        launchKernel(KernelAName, HotTrips);
        launchKernel(KernelBName, HotTrips);
        launchKernel(KernelAName, ColdTrips);
        launchKernel(KernelAName, ColdTrips);
      }
    });
  }
  while (Ready.load(std::memory_order_acquire) != Workers)
    std::this_thread::yield();
  Start.store(true, std::memory_order_release);
  for (auto &Thread : Threads)
    Thread.join();

  check(OccupancyQueries == 2 + Workers * TripsPerWorker,
        "expected one occupancy query per kernel and trip count");
  check(KernelLookups == 2, "expected one lookup per kernel");

  check(__kitcuda_get_global_symbol(&Image, "global_a") == GlobalSymbol,
        "unexpected CUDA global address");
  launchKernel(KernelCName, HotTrips);
  check(__kitcuda_get_global_symbol(&Image, "global_b") == GlobalSymbol,
        "unexpected CUDA global address");
  check(KernelLookups == 3, "expected one lookup per kernel");
  check(ModuleLoads == 1, "kernel and global lookups should share the module");
}
