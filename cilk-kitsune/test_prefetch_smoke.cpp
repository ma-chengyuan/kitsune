#include <cstddef>

// These are the kitrt entry points used by the compiler and its allocator.
extern "C" void *__kitcuda_mem_alloc_managed(std::size_t);
extern "C" void __kitcuda_mem_free(void *);
extern "C" void *__kitcuda_mem_gpu_prefetch(void *, void *);
extern "C" void __kitcuda_sync_thread_stream(void *);
std::size_t __kitrt_get_mem_alloc_size(void *, bool *, bool *);

int main() {
  void *first = __kitcuda_mem_alloc_managed(1);
  // After alignment this allocation ends exactly at the 4 KiB chunk boundary.
  void *last = __kitcuda_mem_alloc_managed(4080);
  bool read_only, write_only;
  if (__kitrt_get_mem_alloc_size(last, &read_only, &write_only) != 4080)
    return 1;
  __kitcuda_sync_thread_stream(__kitcuda_mem_gpu_prefetch(last, nullptr));
  __kitcuda_mem_free(last);
  __kitcuda_mem_free(first);
  return 0;
}
