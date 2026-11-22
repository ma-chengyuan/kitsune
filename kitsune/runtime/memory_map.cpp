//===- memory_map.cpp - Kitsune runtime high-level memory map ---------------===//
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
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

#include "memory_map.h"
#include "kitrt.h"

#include <cassert>
#include <cstdio>
#include <mutex>
#include <new>

#include <boost/intrusive/set.hpp>

using namespace boost::intrusive;

struct KitRTAllocMapEntry : public set_base_hook<optimize_size<true>> {
  void *base;      // base address of the allocation.
  size_t size;     // size of the allocated buffer in bytes.
  bool prefetched; // has the data been prefetched?
  bool read_only;  // upcoming data usage is ("mostly") read only.
  bool write_only; // upcoming data usage is ("mostly") write only.

  friend bool operator<(const KitRTAllocMapEntry &a,
                        const KitRTAllocMapEntry &b) {
    return a.base > b.base; // REVERSE order for splay tree
  }
  friend bool operator==(const KitRTAllocMapEntry &a,
                         const KitRTAllocMapEntry &b) {
    return a.base == b.base;
  }
  friend bool operator!=(const KitRTAllocMapEntry &a,
                         const KitRTAllocMapEntry &b) {
    return a.base != b.base;
  }
};

// Metadata is ordered by decreasing base address.
typedef set<KitRTAllocMapEntry> KitRTAllocMap;
static KitRTAllocMap _kitrt_alloc_map;
static std::shared_mutex _kitrt_alloc_map_mutex;

static KitRTAllocMapEntry *__kitrt_memory_map_lookup(void *addr) {
  // Find the highest base address that is less than or equal to the given
  // address. The decreasing order makes lower_bound a bit tricky to reason
  // about.
  struct Comparator {
    bool operator()(void *addr, const KitRTAllocMapEntry &e) const {
      return addr > e.base;
    }
    bool operator()(const KitRTAllocMapEntry &e, void *addr) const {
      return e.base > addr;
    }
  };
  auto it = _kitrt_alloc_map.lower_bound(addr, Comparator());
  if (it == _kitrt_alloc_map.end()) {
    return nullptr;
  }
  if (reinterpret_cast<uintptr_t>(addr) >=
      reinterpret_cast<uintptr_t>(it->base) + it->size) {
    return nullptr;
  }
  return &(*it);
}

void __kitrt_register_mem_alloc(void *addr, size_t size) {
  assert(addr != nullptr && "unexpected null pointer!");
  // Construct the entry in place so its intrusive set hook starts unlinked,
  // as Boost.Intrusive requires before insertion.
  KitRTAllocMapEntry *entry =
      new (malloc(sizeof(KitRTAllocMapEntry))) KitRTAllocMapEntry();
  entry->base = addr;
  entry->size = size;
  entry->prefetched = false;
  entry->read_only = false;
  entry->write_only = false;

  _kitrt_alloc_map_mutex.lock();
  _kitrt_alloc_map.insert(*entry);
  _kitrt_alloc_map_mutex.unlock();

  if (__kitrt_verbose_mode())
    fprintf(stderr, "kitrt: registered memory allocation (%p) "
	    "of %ld bytes.\n", addr, size);
}

void __kitrt_set_mem_prefetch(void *addr, bool prefetched) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    entry->prefetched = prefetched;
    _kitrt_alloc_map_mutex.unlock_shared();
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitrt: marked memory at %p, size %ld, as '%s'.\n", addr,
              entry->size, prefetched ? "prefetched" : "not prefetched");
  } else {
    _kitrt_alloc_map_mutex.unlock_shared();
  }
  // We could consider a diagnostic here reporting use of an unregistered
  // pointer.  However, this is tricky with the compiler generating calls
  // as it currently has no way to distinguish managed pointer types.  At
  // present we quietly ignore bogus requests...
  //
  // TODO: Should we introduce something more specific at the langauge
  // level to denote managed memory?
}

void __kitrt_mark_mem_read_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    entry->read_only = true;
  }
  _kitrt_alloc_map_mutex.unlock_shared();
}

bool __kitrt_is_mem_read_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  bool ret = entry != nullptr ? entry->read_only : false;
  _kitrt_alloc_map_mutex.unlock_shared();
  return ret;
}

/// @brief Flag the given memory allocation as write only.
/// @param addr: the pointer to the managed memory allocation. 
extern void __kitrt_mark_mem_write_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    entry->write_only = true;
  }
  _kitrt_alloc_map_mutex.unlock_shared();
}


bool __kitrt_is_mem_write_only(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  bool ret = entry != nullptr ? entry->write_only : false;
  _kitrt_alloc_map_mutex.unlock_shared();
  return ret;
}

void __kitrt_clear_mem_advice(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    entry->read_only = false;
    entry->write_only = false;
  }
  _kitrt_alloc_map_mutex.unlock_shared();
}

bool __kitrt_is_mem_prefetched(void *addr, size_t *size, void **base) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  bool prefetched = true;
  if (entry != nullptr) {
    if (size != nullptr)
      *size = entry->size;
    if (base != nullptr)
      *base = entry->base;
    prefetched = entry->prefetched;
  } else {
    // NOTE: This is a bit strange but we have to deal with the
    // compiler's code generation mechanisms here.  Specifically it is
    // only able to identify pointers but not pointers allocated in
    // managed memory.  For this reason we may generate requests for
    // un-managed pointers.  To "behave" we currently treat such
    // requests as prefetched (and thus avoid additional calls that
    // might presume a valid managed memory region is associated with
    // the pointer).
    prefetched = true;
  }
  _kitrt_alloc_map_mutex.unlock_shared();
  return prefetched;
}

size_t __kitrt_get_mem_alloc_size(void *addr, bool *read_only,
                                  bool *write_only) {
  assert(addr != nullptr && "unexpected null addr pointer!");
  assert(read_only != nullptr && "unexpected null read_only pointer!");
  assert(write_only != nullptr && "unexpected null write_only pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  size_t size = 0;
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr && addr == entry->base) {
    *read_only = entry->read_only;
    *write_only = entry->write_only;
    size = entry->size;
  } else {
    *read_only = false;
    *write_only = false;
  }
  _kitrt_alloc_map_mutex.unlock_shared();

  return size;
}

void __kitrt_unregister_mem_alloc(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    _kitrt_alloc_map.erase(*entry);
    entry->~KitRTAllocMapEntry();
    free(entry);
  }
  _kitrt_alloc_map_mutex.unlock();
}

void __kitrt_mem_needs_prefetch(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  _kitrt_alloc_map_mutex.lock_shared();
  KitRTAllocMapEntry *entry = __kitrt_memory_map_lookup(addr);
  if (entry != nullptr) {
    entry->prefetched = false;
  }
  _kitrt_alloc_map_mutex.unlock_shared();
}

extern "C" void __kitrt_print_memory_map() {
  fprintf(stdout, "kitsune runtime memory allocation map:\n");
  _kitrt_alloc_map_mutex.lock();
  if (_kitrt_alloc_map.empty()) 
    fprintf(stdout, "\t[... empty ...]\n");
  else {
    const size_t MBYTE = 1024 * 1024;
    size_t total_allocated = 0;
    unsigned int num_allocations = 0;
    for(auto &entry : _kitrt_alloc_map) {
      const KitRTAllocMapEntry *alloc_entry = &entry;
      total_allocated += alloc_entry->size;
      num_allocations++;
      fprintf(stderr,
              "\tAddress: %p --> [size: %6.2f Mbytes, prefetched: %8s, "
              "read-only: %8s, write-only: %8s]\n",
              entry.base, alloc_entry->size / (double)MBYTE,
              alloc_entry->prefetched ? "true" : "false",
              alloc_entry->read_only ? "true" : "false",
              alloc_entry->write_only ? "true" : "false");
    }
    fprintf(stderr, "\n");
    fprintf(stdout, "\ttotal memory allocation: %6.2f Mbytes\n",
	    total_allocated / (double)MBYTE);
    fprintf(stderr, "\taverage size per allocation: %6.2f Mbytes\n", 
            (total_allocated / (double)MBYTE)/ num_allocations);
  }
  _kitrt_alloc_map_mutex.unlock();
}

extern "C" void __kitrt_destroy_memory_map(void (*free_mem_call)(void *)) {
  assert(free_mem_call != nullptr && "unexpected null function pointer!");
  _kitrt_alloc_map_mutex.lock();
  kitrt::vector<KitRTAllocMapEntry *> entries(_kitrt_alloc_map.size());
  fprintf(stderr, "kitrt: destroying memory map, freeing %zu entries.\n",
          _kitrt_alloc_map.size());
  for (auto &entry : _kitrt_alloc_map) {
    free_mem_call(entry.base);
    entries.push_back(&entry);
  }
  _kitrt_alloc_map.clear();
  for (auto &entry : entries) {
    free(entry);
  }
  _kitrt_alloc_map_mutex.unlock();
}

