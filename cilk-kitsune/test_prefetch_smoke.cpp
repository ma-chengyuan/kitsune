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

static bool check_prefetch(void *ptr, size_t size, CUdevice gpu,
                           CUdevice expected_mapping) {
  if (!check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, gpu) ||
      !check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      CU_DEVICE_INVALID))
    return false;

  __kitcuda_sync_thread_stream(__kitcuda_mem_gpu_prefetch(ptr, nullptr));
  if (!check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
                      gpu) ||
      !check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
                      expected_mapping) ||
      !check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                      CU_DEVICE_INVALID))
    return false;

  __kitcuda_sync_thread_stream(__kitcuda_mem_host_prefetch(ptr, nullptr));
  return check_location(ptr, size,
                        CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
                        CU_DEVICE_CPU) &&
         check_location(ptr, size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                        CU_DEVICE_INVALID);
}

int main() {
  void *first = __kitcuda_mem_alloc_managed(1);
  // After alignment this allocation ends exactly at the 4 KiB chunk boundary.
  void *last = __kitcuda_mem_alloc_managed(4080);
  // At the pool threshold the allocator gives the buffer its own pages.
  void *dedicated = __kitcuda_mem_alloc_managed(4096);
  bool read_only, write_only;
  if (__kitrt_get_mem_alloc_size(last, &read_only, &write_only) != 4080)
    return 1;
  CUdevice gpu;
  if (cuCtxGetDevice(&gpu) != CUDA_SUCCESS ||
      !check_prefetch(last, 4080, gpu, gpu) ||
      !check_prefetch(dedicated, 4096, gpu, CU_DEVICE_INVALID))
    return 1;
  __kitcuda_mem_free(dedicated);
  __kitcuda_mem_free(last);
  __kitcuda_mem_free(first);
  return 0;
}
