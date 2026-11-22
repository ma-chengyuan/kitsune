//===- memory.cpp - Kitsune runtime CUDA memory support  ------------------===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This software
//  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
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
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include "memory_map.h"

#include <atomic>
#include <cstdio>
#include <mutex>

static std::mutex _kitcuda_mem_alloc_mutex;
static kitrt::unordered_set<void *> _kitcuda_big_blocks;

size_t _kitcuda_big_block_threshold = 4096;
size_t _kitcuda_memops_threshold = 4096;

struct ThreadLocalData {
  uintptr_t cur_block = 0;
  size_t capacity_remaining = 0;
};

static thread_local ThreadLocalData _kitcuda_tls_data;

struct Chunk {
  void *ptr;
  Chunk *next;
};
static std::atomic<Chunk *> _kitcuda_chunk_list{nullptr};
static std::atomic_size_t _kitcuda_chunk_fragmentation{0};

// Reducer cache

struct ReducerCacheSizeClass {
  std::mutex mutex;
  kitrt::vector<void *> cache;
};

static constexpr size_t N_REDUCER_CACHE_SIZE_CLASSES = 32;
static ReducerCacheSizeClass
    _kitcuda_reducer_cache[N_REDUCER_CACHE_SIZE_CLASSES];

extern "C" {

static __attribute__((malloc)) void *
__kitcuda_mem_alloc_managed_internal(size_t size) {
  KIT_NVTX_PUSH("kitcuda:mem_alloc_managed", KIT_NVTX_MEM);
  __kitcuda_set_context();

  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_GLOBAL));

  // Shared allocations may alternate between CPU and GPU phases. Allow GPU
  // access and leave the preferred location to the allocation's owner.
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_ACCESSED_BY,
                             _kitcuda_device));

  int enable = 1;
  CU_SAFE_CALL(
      cuPointerSetAttribute_p(&enable, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devp));
  KIT_NVTX_POP();

  // NOTE: We can no longer do this in a thread-safe manner...
  // CU_SAFE_CALL(cuMemPrefetchAsync_p(devp, size, _kitcuda_device,
  //                                  __kitcuda_get_thread_stream()));
  return reinterpret_cast<void *>(devp);
}

__attribute__((malloc)) void *
__kitcuda_mem_alloc_managed_aligned(size_t size, size_t alignment) {

  extern bool _kitcuda_initialized;
  if (not _kitcuda_initialized)
    __kitcuda_initialize();

  void *ret = nullptr;

  if (size < _kitcuda_big_block_threshold) {
    ThreadLocalData &tls_data = _kitcuda_tls_data;
    size_t to_align =
        ((tls_data.cur_block + alignment - 1) & ~(alignment - 1)) -
        tls_data.cur_block;
    if (to_align > tls_data.capacity_remaining ||
        size > tls_data.capacity_remaining - to_align) {
      _kitcuda_chunk_fragmentation += tls_data.capacity_remaining;
      tls_data.cur_block = reinterpret_cast<uintptr_t>(
          __kitcuda_mem_alloc_managed_internal(_kitcuda_big_block_threshold));
      Chunk *new_chunk = reinterpret_cast<Chunk *>(malloc(sizeof(Chunk)));
      // ^ Don't use new because new might have been replaced to use this
      // library and we might end up with infinite recursion.
      new_chunk->ptr = reinterpret_cast<void *>(tls_data.cur_block);
      // Insert the new chunk to our chunklist via CAS
      Chunk *next = _kitcuda_chunk_list.load(std::memory_order_acquire);
      do {
        new_chunk->next = next;
      } while (!_kitcuda_chunk_list.compare_exchange_strong(
          next, new_chunk, std::memory_order_acq_rel,
          std::memory_order_acquire));
      tls_data.capacity_remaining = _kitcuda_big_block_threshold;
      to_align = 0;
    }
    ret = reinterpret_cast<void *>(tls_data.cur_block + to_align);
    // Padding precedes the user pointer and is not part of its prefetch range.
    tls_data.cur_block += to_align + size;
    tls_data.capacity_remaining -= to_align + size;
    _kitcuda_mem_alloc_mutex.lock();
  } else {
    ret = reinterpret_cast<void *>(__kitcuda_mem_alloc_managed_internal(size));
    _kitcuda_mem_alloc_mutex.lock();
    _kitcuda_big_blocks.insert(ret);
  }

  // Register this allocation so the runtime can help track the
  // locality (and affinity) of data.
  __kitrt_register_mem_alloc(ret, size);
  _kitcuda_mem_alloc_mutex.unlock();
  return ret;
}

__attribute__((malloc)) void *__kitcuda_mem_alloc_managed(size_t size) {
  return __kitcuda_mem_alloc_managed_aligned(size,
                                             __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

__attribute__((malloc)) void *
__kitcuda_mem_calloc_managed(size_t count, size_t element_size) {
  assert(count != 0 && "zero-valued item count!");
  assert(element_size != 0 && "zero-valued element size!");

  KIT_NVTX_PUSH("kitcuda:calloc_managed", KIT_NVTX_MEM);

  size_t nbytes = count * element_size;
  CUdeviceptr memp = (CUdeviceptr)__kitcuda_mem_alloc_managed(nbytes);

  // TODO: Is there a risk of a race here?  From the driver API docs:
  //
  //   The cudaMemset functions are asynchronous with respect to the host
  //   except when the target memory is pinned host memory. The Async
  //   versions are always asynchronous with respect to the host.
  //
  // Given our use of UVM we might also be able to use a straight memset()
  // call... which would, of course, place all pages on the host...
  //
  // TODO: We're not set to run on anything but the default stream...
  CU_SAFE_CALL(cuMemsetD8Async_p(memp, 0, nbytes, NULL));
  KIT_NVTX_POP();
  return (void *)memp;
}

__attribute__((malloc)) void *__kitcuda_mem_realloc_managed(void *ptr,
                                                            size_t size) {
  assert(size != 0 && "zero-valued size!");

  KIT_NVTX_PUSH("kitcuda:realloc_managed", KIT_NVTX_MEM);
  void *memptr = nullptr;
  if (ptr == nullptr)
    memptr = __kitcuda_mem_alloc_managed(size);
  else {
    // Check to make sure this is a pointer we're actually managing.
    bool read_only, write_only;
    size_t nbytes = __kitrt_get_mem_alloc_size(ptr, &read_only, &write_only);
    if (nbytes == 0) {
      fprintf(stderr, "kitcuda: warning, realloc() on untracked allocation!\n");
      KIT_NVTX_POP();
      return nullptr;
    }

    if (size > nbytes) {
      // Requested size is larger than currently tracked allocation.
      // Replace it.
      memptr = __kitcuda_mem_alloc_managed(size);
      cuMemcpy_p(/* dest */ (CUdeviceptr)memptr,
                 /* source */ (CUdeviceptr)ptr, nbytes);

      // NOTE: realloc does not guarantee initialized memory outside
      // of existing data...
      __kitcuda_mem_free(ptr);
    } else if (size < nbytes) {
      memptr = __kitcuda_mem_alloc_managed(size);
      cuMemcpy_p(/* dest */ (CUdeviceptr)memptr,
                 /* source */ (CUdeviceptr)ptr, size);
      __kitcuda_mem_free(ptr);
    } else
      memptr = ptr; // same size, just return it...
  }
  KIT_NVTX_POP();
  return memptr;
}

void __kitcuda_mem_free(void *vp) {
  assert(vp && "unexpected null pointer!");

  // We first remove the allocation from the runtime's
  // map, and then actually release it via CUDA...
  // Note that the versioned free calls are important
  // here -- a non-v2 version will actually result in
  // crashes...

  bool is_big_block = false;
  _kitcuda_mem_alloc_mutex.lock();
  __kitrt_unregister_mem_alloc(vp);
  if (auto it = _kitcuda_big_blocks.find(vp); it != _kitcuda_big_blocks.end()) {
    is_big_block = true;
    _kitcuda_big_blocks.erase(it);
  }
  _kitcuda_mem_alloc_mutex.unlock();
  if (is_big_block) {
    KIT_NVTX_PUSH("kitcuda:mem_free", KIT_NVTX_MEM);
    CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
    KIT_NVTX_POP();
  }
}

void __kitcuda_mem_destroy(void *vp) {
  KIT_NVTX_PUSH("kitcuda: mem_destroy", KIT_NVTX_MEM);
  bool is_big_block = false;
  _kitcuda_mem_alloc_mutex.lock();
  if (auto it = _kitcuda_big_blocks.find(vp); it != _kitcuda_big_blocks.end()) {
    is_big_block = true;
    _kitcuda_big_blocks.erase(it);
  }
  _kitcuda_mem_alloc_mutex.unlock();
  if (is_big_block) {
    // This entry point is used to clean up only the
    // CUDA portions of an allocation -- it is used
    // by the runtime at program exit.
    // CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
    if (cuMemFree_v2_p((CUdeviceptr)vp) != CUDA_SUCCESS) {
      fprintf(stderr, "kitcuda: warning, failed to free memory: %p\n", vp);
    }
  }
  KIT_NVTX_POP();
}

void __kitcuda_destroy_mem_chunks() {
  size_t total_alloc = 0;
  for (Chunk *chunk = _kitcuda_chunk_list.load(); chunk;) {
    Chunk *next = chunk->next;
    total_alloc += _kitcuda_big_block_threshold;
    CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)chunk->ptr));
    free(chunk);
    chunk = next;
  }
  _kitcuda_chunk_list.store(nullptr);
  fprintf(stderr,
          "kitcuda: freed all memory chunks, fragmentation: %zu / %zu = %.3f\n",
          _kitcuda_chunk_fragmentation.load(), total_alloc,
          static_cast<double>(_kitcuda_chunk_fragmentation.load()) /
              total_alloc);
  _kitcuda_chunk_fragmentation.store(0);
}

bool __kitcuda_is_mem_managed(void *vp) {
  assert(vp && "unexpected null pointer!");
  assert(__kitcuda_is_initialized() && "kitrt: runtime not initialized!");
  KIT_NVTX_PUSH("kitcuda:is_mem_managed", KIT_NVTX_MEM);

  CUdeviceptr devp = (CUdeviceptr)vp;
  unsigned int is_managed;
  // NOTE: We don't wrap in a CUDA-safe call here as we could be
  // passing a bogus pointer -- if we get a CUDA error we will
  // assume the pointer is unmanaged and return false accordingly.
  CUresult r = cuPointerGetAttribute_p(&is_managed,
                                       CU_POINTER_ATTRIBUTE_IS_MANAGED, devp);
  KIT_NVTX_POP();
  return (r == CUDA_SUCCESS) && is_managed;
}

// NOTE: See within the code below for notes about the prefetching
// semantics.
void *__kitcuda_mem_gpu_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "unexpected null pointer!");

  KIT_NVTX_PUSH("kitcuda:mem_gpu_prefetch", 1);

  size_t size = 0;
  void *base = nullptr;
  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis
  // and code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior yet.  If we have
  // ever issued a prefetch on a data region it will show here as
  // prefetched and we avoid reissuing a prefetch.  There are
  // obviously cases where this is helpful and others where it will
  // lead to page faults and evictions of pages...  At present this
  // has lead to the best general performance and reduced complexity,
  // while also maintaining correctness.
  if (not __kitrt_is_mem_prefetched(vp, &size, &base)) {
    if (size > 0) {
      __kitcuda_set_context();
      vp = base;

      // Issue a prefetch request on the provided stream.  If the given
      // stream is null, create a new stream and return it. Once issued
      // go ahead and mark the memory as having been prefetched.  This
      // "mark" does not guarantee prefetching is complete it simply
      // flags that the "instruction" has been issued by the runtime.
      CUstream cu_stream;
      if (opaque_stream)
        cu_stream = (CUstream)opaque_stream;
      else
        cu_stream = (CUstream)__kitcuda_get_thread_stream();

      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, _kitcuda_device,
                                        cu_stream));
      __kitrt_mark_mem_prefetched(vp);
      KIT_NVTX_POP();
      return (void *)cu_stream;
    }
  } /*else if (!__kitcuda_is_mem_managed(vp)) {
    // We are cooked
    fprintf(stderr, "kitcuda: warning, prefetching unmanaged memory: %p\n", vp);
  }*/
  KIT_NVTX_POP();
  // no prefetch, no bound stream to bound it to...
  return nullptr;
}

void *__kitcuda_mem_host_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "unexpected null pointer!");

  KIT_NVTX_PUSH("kitcuda:mem_host_prefetch", KIT_NVTX_MEM);

  size_t size;
  void *base;
  if (__kitrt_is_mem_prefetched(vp, &size, &base)) {
    if (size > 0) {
      vp = base;
      // Issue a prefetch request on the stream associated with the
      // calling thread. Once issued go ahead and mark the memory as
      // no long being prefetched to the device/GPU.  This "mark" does
      // not guarantee prefetching is complete it simply flags that
      // the "instruction" has been issued by the runtime.
      CUstream cu_stream;
      if (opaque_stream)
        cu_stream = (CUstream)opaque_stream;
      else
        cu_stream = (CUstream)__kitcuda_get_thread_stream();

      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, CU_DEVICE_CPU,
                                        cu_stream));
      __kitrt_set_mem_prefetch(vp, false);
      KIT_NVTX_POP();
      return cu_stream;
    }
  }
  KIT_NVTX_POP();
  return nullptr;
}

void __kitcuda_memcpy_sym_to_device(void *hostPtr, uint64_t devPtr,
                                    size_t size) {
  assert(devPtr != 0 && "unexpected null device pointer!");
  assert(hostPtr != nullptr && "unexpected null host pointer!");
  assert(size != 0 && "requested a 0 byte copy!");

  KIT_NVTX_PUSH("kitcuda:memcpy_sym_to_device", KIT_NVTX_MEM);
  CU_SAFE_CALL(cuMemcpyHtoD_v2_p(devPtr, hostPtr, size));
  KIT_NVTX_POP();
}

void *__kitcuda_mem_alloc_and_copy_to_device(void *host_ptr, size_t size,
                                             void **dev_ptr,
                                             void *opaque_stream) {
  __kitcuda_set_context();

  KIT_NVTX_PUSH("kitcuda:mem_alloc_device_and_copy", KIT_NVTX_MEM);
  CUstream cu_stream = opaque_stream ? (CUstream)opaque_stream
                                     : (CUstream)__kitcuda_get_thread_stream();

  const size_t size_class =
      64 - __builtin_clzll(size - 1); // Round up to nearest power of 2

  if (size_class < N_REDUCER_CACHE_SIZE_CLASSES) {
    ReducerCacheSizeClass &reducer_cache = _kitcuda_reducer_cache[size_class];
    reducer_cache.mutex.lock();
    if (!reducer_cache.cache.empty()) {
      *dev_ptr = reducer_cache.cache.back();
      reducer_cache.cache.pop_back();
      reducer_cache.mutex.unlock();
    } else {
      reducer_cache.mutex.unlock();
      CU_SAFE_CALL(cuMemAlloc_v2_p((CUdeviceptr *)dev_ptr, 1L << size_class));
    }
  } else {
    // I don't think this is ever going to happen, but just in case...
    CU_SAFE_CALL(cuMemAlloc_v2_p((CUdeviceptr *)dev_ptr, size));
  }

  CU_SAFE_CALL(
      cuMemcpyHtoDAsync_v2_p((CUdeviceptr)*dev_ptr, host_ptr, size, cu_stream));
  KIT_NVTX_POP();
  return cu_stream;
}

void *__kitcuda_mem_copy_and_free_from_device(void *host_ptr, size_t size,
                                              void *dev_ptr,
                                              void *opaque_stream) {
  // Skip context check because we assume __kitcuda_mem_alloc_and_copy_to_device
  // was called before.
  KIT_NVTX_PUSH("kitcuda:mem_copy_device_and_free", KIT_NVTX_MEM);
  CUstream cu_stream = opaque_stream ? (CUstream)opaque_stream
                                     : (CUstream)__kitcuda_get_thread_stream();
  CU_SAFE_CALL(
      cuMemcpyDtoHAsync_v2_p(host_ptr, (CUdeviceptr)dev_ptr, size, cu_stream));

  const size_t size_class =
      64 - __builtin_clzll(size - 1); // Round up to nearest power of 2
  if (size_class < N_REDUCER_CACHE_SIZE_CLASSES) {
    ReducerCacheSizeClass &reducer_cache = _kitcuda_reducer_cache[size_class];
    reducer_cache.mutex.lock();
    reducer_cache.cache.push_back(dev_ptr);
    reducer_cache.mutex.unlock();
  } else {
    CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)dev_ptr));
  }

  KIT_NVTX_POP();
  return cu_stream;
}

void __kitcuda_destroy_reducer_cache() {
  for (size_t i = 0; i < 32; ++i) {
    ReducerCacheSizeClass &reducer_cache = _kitcuda_reducer_cache[i];
    reducer_cache.mutex.lock();
    for (void *ptr : reducer_cache.cache) {
      CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)ptr));
    }
    reducer_cache.cache.clear();
    reducer_cache.mutex.unlock();
  }
}

void __kitcuda_memcpy(void *dst, void *src, size_t size) {
  // For now, just use the host memcpy.
  if (size < _kitcuda_memops_threshold) {
    // Small memcpy, use the host memcpy anyways because a pagefault doesn't
    // hurt as much.
    memcpy(dst, src, size);
    return;
  }
  extern bool _kitcuda_initialized;
  if (not _kitcuda_initialized)
    __kitcuda_initialize();

  if (!__kitcuda_is_mem_managed(dst) || !__kitcuda_is_mem_managed(src)) {
    // At least one of dst or src is unmanaged, use the host memcpy.
    memcpy(dst, src, size);
    return;
  }
  __kitcuda_set_context();

  CU_SAFE_CALL(cuMemcpy_p(reinterpret_cast<CUdeviceptr>(dst),
                          reinterpret_cast<CUdeviceptr>(src), size));
}

void __kitcuda_memmove(void *dst, void *src, size_t size) {
  const auto dst_ = reinterpret_cast<uintptr_t>(dst);
  const auto src_ = reinterpret_cast<uintptr_t>(src);
  if (dst_ < src_ + size && src_ < dst_ + size) {
    // Memory regions overlap, use memmove because cuMemCpy doesn't really
    // handle this well. Page faults are the price we have to pay for such edge
    // cases (meh).
    memmove(dst, src, size);
    return;
  }
  // No overlap, use our optimized memcpy implementation
  __kitcuda_memcpy(dst, src, size);
}

void __kitcuda_memset(void *dst, uint8_t value, size_t size) {
  if (size < _kitcuda_memops_threshold) {
    memset(dst, value, size);
    return;
  }

  extern bool _kitcuda_initialized;
  if (not _kitcuda_initialized)
    __kitcuda_initialize();

  __kitcuda_set_context();

  if (__kitcuda_is_mem_managed(dst)) {
    CU_SAFE_CALL(
        cuMemsetD8_v2_p(reinterpret_cast<CUdeviceptr>(dst), value, size));
  } else {
    memset(dst, value, size);
  }
}
}
