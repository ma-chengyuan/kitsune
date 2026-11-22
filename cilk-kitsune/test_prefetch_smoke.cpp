#include <cstddef>
#include <cstdio>
#include <cuda.h>

// These are the kitrt entry points used by the compiler and its allocator.
extern "C" void *__kitcuda_mem_alloc_managed(std::size_t);
extern "C" void __kitcuda_mem_free(void *);
extern "C" void *__kitcuda_mem_gpu_prefetch(void *, void *);
extern "C" void *__kitcuda_mem_host_prefetch(void *, void *);
extern "C" void __kitcuda_sync_thread_stream(void *);
std::size_t __kitrt_get_mem_alloc_size(void *, bool *, bool *);

static bool check_location(void *ptr, size_t size, CUmem_range_attribute attr,
                           CUdevice expected) {
  CUdevice value = CU_DEVICE_INVALID;
  CUresult status = cuMemRangeGetAttribute(
      &value, sizeof(value), attr, reinterpret_cast<CUdeviceptr>(ptr), size);
  if (status == CUDA_SUCCESS && value == expected)
    return true;
  std::fprintf(stderr,
               "CUDA range attribute %d: expected %d, got %d (status %d)\n",
               attr, expected, value, status);
  return false;
}

int main() {
  void *first = __kitcuda_mem_alloc_managed(1);
  // After alignment this allocation ends exactly at the 4 KiB chunk boundary.
  void *last = __kitcuda_mem_alloc_managed(4080);
  bool read_only, write_only;
  if (__kitrt_get_mem_alloc_size(last, &read_only, &write_only) != 4080)
    return 1;
  if (!check_location(last, 4080, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      CU_DEVICE_INVALID))
    return 1;
  CUdevice gpu;
  if (cuCtxGetDevice(&gpu) != CUDA_SUCCESS)
    return 1;
  __kitcuda_sync_thread_stream(__kitcuda_mem_gpu_prefetch(last, nullptr));
  if (!check_location(last, 4080, CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
                      gpu) ||
      !check_location(last, 4080, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      CU_DEVICE_INVALID))
    return 1;
  __kitcuda_sync_thread_stream(__kitcuda_mem_host_prefetch(last, nullptr));
  if (!check_location(last, 4080, CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
                      CU_DEVICE_CPU) ||
      !check_location(last, 4080, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      CU_DEVICE_INVALID))
    return 1;
  __kitcuda_mem_free(last);
  __kitcuda_mem_free(first);
  return 0;
}
