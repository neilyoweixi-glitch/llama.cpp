# Pool Allocation Algorithm Analysis: CUDA vs CANN

This document provides a detailed comparison of the memory pooling algorithms used in CUDA and CANN backends.

## Overview

Both CUDA and CANN implement multiple pool strategies:
- **Legacy/Buffer pools**: Fixed-size arrays with reuse logic
- **Priority queue pools**: Heap-based selection (CANN only)
- **Virtual Memory (VMM) pools**: Stack-based allocation with virtual memory mapping

## Algorithm Comparison Matrix

| Feature | CUDA Legacy | CUDA VMM | CANN Buffer | CANN Priority | CANN VMM |
|---------|-------------|----------|-------------|----------------|----------|
| **Data Structure** | Fixed array (256) | Stack pointer | Fixed array (256) | Priority queue + HashMap | Stack pointer |
| **Selection Strategy** | Best-fit (min diff) | Stack (LIFO) | First-fit | Smallest-fit (min heap) | Stack (LIFO) |
| **Reuse Logic** | Exact match preferred | No reuse | Margin-based (4MB) | Margin-based (4MB) | No reuse |
| **Cleanup Strategy** | None | None | Time-based (100ms) | Time-based (100ms) | None |
| **Allocation Overhead** | O(n) scan | O(1) | O(n) scan | O(log n) heap ops | O(1) |
| **Free Overhead** | O(n) scan | O(1) | O(n) scan | O(log n) insert | O(1) |
| **Memory Growth** | 5% lookahead | Granularity-based | Exact | Exact | Granularity-based |
| **Max Pool Size** | Unlimited (256 slots) | 32 GB | Unlimited (256 slots) | Unlimited | Device VRAM |

---

## 1. Legacy/Buffer Pool Algorithms

### CUDA Legacy Pool (`ggml_cuda_pool_leg`)

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:316-411`

#### Allocation Algorithm

```c
void * alloc(size_t size, size_t * actual_size) {
    // 1. Best-fit search: Find buffer with smallest size >= requested size
    size_t best_diff = 1ull << 36;
    int ibest = -1;
    
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        if (buffer_pool[i].ptr != nullptr && buffer_pool[i].size >= size) {
            size_t diff = buffer_pool[i].size - size;
            if (diff < best_diff) {
                best_diff = diff;
                ibest = i;
                if (diff == 0) break;  // Early exit on exact match
            }
        }
    }
    
    // 2. Reuse if found
    if (ibest >= 0) {
        return buffer_pool[ibest].ptr;  // Remove from pool
    }
    
    // 3. Allocate new with 5% lookahead + 256-byte alignment
    size_t look_ahead_size = (size_t)(1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255) / 256);
    cudaMalloc(&ptr, look_ahead_size);
    return ptr;
}
```

**Characteristics**:
- **Selection**: Best-fit (minimizes waste)
- **Time Complexity**: O(n) where n = MAX_BUFFERS (256)
- **Memory Growth**: 5% lookahead + 256-byte alignment
- **Reuse**: Exact match preferred, otherwise best-fit
- **No cleanup**: Buffers accumulate until pool is full

#### Free Algorithm

```c
void free(void * ptr, size_t size) {
    // Linear search for empty slot
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        if (buffer_pool[i].ptr == nullptr) {
            buffer_pool[i].ptr = ptr;
            buffer_pool[i].size = size;
            return;
        }
    }
    // Pool full: free immediately
    cudaFree(ptr);
}
```

**Characteristics**:
- **Time Complexity**: O(n) worst case, O(1) best case (first empty slot)
- **No ordering**: Buffers stored in first available slot
- **Overflow handling**: Frees immediately if pool is full

---

### CANN Buffer Pool (`ggml_cann_pool_buf`)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:398-592`

#### Allocation Algorithm

```cpp
void * alloc(size_t size, size_t * actual_size) {
    size = GGML_PAD(size, 128);  // 128-byte alignment
    
    void * ptr = nullptr;
    auto now = std::chrono::steady_clock::now();
    
    // 1. First-fit search with cleanup
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        if (buffer_pool[i].ptr == nullptr) break;  // Found empty slot
        if (buffer_pool[i].used) continue;  // Skip in-use buffers
        
        if (buffer_pool[i].size >= size) {
            size_t margin = buffer_pool[i].size - size;
            if (margin <= max_reuse_margin) {  // 4MB threshold
                // Reuse buffer
                buffer_pool[i].used = true;
                return buffer_pool[i].ptr;
            }
        }
        
        // 2. Cleanup old buffers (>100ms unused, >1MB)
        bool should_clean = !disable_clean && 
                           buffer_pool[i].size > min_free_margin &&
                           (now - buffer_pool[i].last_used) > 100ms;
        if (should_clean) {
            aclrtFree(buffer_pool[i].ptr);
            buffer_pool[i].ptr = nullptr;  // Mark as empty
        }
    }
    
    // 3. Allocate new buffer
    if (i < MAX_BUFFERS) {
        aclrtMalloc(&buffer_pool[i].ptr, size);
        buffer_pool[i].size = size;
        buffer_pool[i].used = true;
        return buffer_pool[i].ptr;
    }
    
    ABORT("Pool full");
}
```

**Characteristics**:
- **Selection**: First-fit with margin tolerance (4MB)
- **Time Complexity**: O(n) scan with cleanup
- **Memory Growth**: Exact size (no lookahead)
- **Reuse**: Margin-based (accepts buffers up to 4MB larger)
- **Cleanup**: Time-based eviction (>100ms unused, >1MB)

#### Free Algorithm

```cpp
void free(void * ptr, size_t size) {
    // Linear search by pointer
    for (int i = 0; i < MAX_BUFFERS; ++i) {
        if (buffer_pool[i].ptr == ptr) {
            buffer_pool[i].used = false;
            buffer_pool[i].last_used = std::chrono::steady_clock::now();
            return;
        }
    }
    ABORT("Buffer not found");
}
```

**Characteristics**:
- **Time Complexity**: O(n) linear search
- **State tracking**: Marks buffer as unused, updates timestamp
- **No reordering**: Buffers stay in place

---

## 2. Priority Queue Pool (CANN Only)

### CANN Priority Pool (`ggml_cann_pool_buf_prio`)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:208-391`

#### Data Structures

```cpp
std::unordered_map<void *, size_t> buffer_pool;  // Active buffers
std::priority_queue<ggml_cann_buffer, 
                    std::vector<ggml_cann_buffer>, 
                    std::greater<>> free_buffers;  // Min-heap by size
```

#### Allocation Algorithm

```cpp
void * alloc(size_t size, size_t * actual_size) {
    size = GGML_PAD(size, 128);
    void * ptr = nullptr;
    auto now = std::chrono::steady_clock::now();
    
    std::vector<ggml_cann_buffer> free_buffers_rest;
    
    // 1. Pop from min-heap until suitable buffer found
    while (!free_buffers.empty()) {
        auto b = free_buffers.top();
        free_buffers.pop();
        
        if (b.size >= size) {
            size_t margin = b.size - size;
            if (margin <= max_reuse_margin) {  // 4MB threshold
                // Reuse: smallest buffer that fits
                ptr = b.ptr;
                break;
            }
        }
        
        // 2. Cleanup old buffers
        bool should_clean = !disable_clean && 
                           b.size > min_free_margin &&
                           (now - b.last_used) > 100ms;
        if (should_clean) {
            aclrtFree(b.ptr);
            buffer_pool.erase(b.ptr);
            continue;  // Don't push back
        }
        
        free_buffers_rest.push_back(b);
    }
    
    // 3. Push remaining buffers back
    for (auto & b : free_buffers_rest) {
        free_buffers.push(b);
    }
    
    // 4. Allocate new if needed
    if (ptr == nullptr) {
        aclrtMalloc(&ptr, size);
        buffer_pool.emplace(ptr, size);
    }
    
    return ptr;
}
```

**Characteristics**:
- **Selection**: Smallest-fit (min-heap ensures smallest suitable buffer)
- **Time Complexity**: O(k log n) where k = buffers examined, n = heap size
- **Memory Growth**: Exact size
- **Reuse**: Margin-based (4MB), prefers smallest buffer
- **Cleanup**: Time-based eviction during allocation scan

#### Free Algorithm

```cpp
void free(void * ptr, size_t size) {
    auto it = buffer_pool.find(ptr);  // O(1) lookup
    if (it == buffer_pool.end()) ABORT("Not found");
    
    auto now = std::chrono::steady_clock::now();
    free_buffers.emplace(ggml_cann_buffer{ptr, it->second, now});  // O(log n) insert
}
```

**Characteristics**:
- **Time Complexity**: O(log n) heap insert + O(1) hash lookup
- **Ordering**: Maintains min-heap property
- **Efficiency**: Better than linear search for large pools

---

## 3. Virtual Memory Pool Algorithms

### CUDA VMM Pool (`ggml_cuda_pool_vmm`)

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:415-521`

#### Allocation Algorithm

```c
void * alloc(size_t size, size_t * actual_size) {
    size = GGML_PAD(size, 128);  // 128-byte alignment
    
    size_t avail = pool_size - pool_used;
    
    // 1. Expand pool if needed
    if (size > avail) {
        size_t reserve_size = size - avail;
        reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);
        
        // Reserve virtual address space (once)
        if (pool_addr == 0) {
            cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0);
        }
        
        // Allocate physical memory
        cuMemCreate(&handle, reserve_size, &prop, 0);
        
        // Map at end of pool
        cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0);
        
        // Set access permissions
        cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1);
        
        pool_size += reserve_size;
    }
    
    // 2. Stack allocation (LIFO)
    void * ptr = (void *)(pool_addr + pool_used);
    pool_used += size;
    return ptr;
}
```

**Characteristics**:
- **Allocation**: Stack-based (LIFO), O(1)
- **Memory Growth**: Granularity-aligned chunks
- **No reuse**: Requires reverse-order deallocation
- **Virtual Memory**: Uses CUDA VMM API for large address space

#### Free Algorithm

```c
void free(void * ptr, size_t size) {
    pool_used -= size;
    // Assert: ptr must equal pool_addr + pool_used (reverse order)
    GGML_ASSERT(ptr == (void *)(pool_addr + pool_used));
}
```

**Characteristics**:
- **Time Complexity**: O(1)
- **Constraint**: Must free in reverse order of allocation
- **No fragmentation**: Stack discipline prevents holes

---

### CANN VMM Pool (`ggml_cann_pool_vmm`)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:600-756`

#### Allocation Algorithm

```cpp
void * alloc(size_t size, size_t * actual_size) {
    size = GGML_PAD(size, 128);
    
    size_t avail = pool_size - pool_used;
    
    // 1. Expand pool if needed
    if (size > avail) {
        size_t reserve_size = size - avail;
        reserve_size = GGML_PAD(reserve_size, granularity);
        
        // Reserve virtual address space (once)
        if (pool_addr == 0) {
            aclrtReserveMemAddress(&pool_addr, max_size, 0, NULL, 1);
        }
        
        // Allocate physical memory
        aclrtMallocPhysical(&handle, reserve_size, &prop, 0);
        
        // Map at end of pool
        aclrtMapMem((char *)pool_addr + pool_size, reserve_size, 0, handle, 0);
        
        handles.push_back(handle);
        map_offsets.push_back((char *)pool_addr + pool_size);
        
        pool_size += reserve_size;
    }
    
    // 2. Stack allocation (LIFO)
    void * ptr = (void *)((char *)pool_addr + pool_used);
    pool_used += size;
    return ptr;
}
```

**Characteristics**:
- **Allocation**: Stack-based (LIFO), O(1)
- **Memory Growth**: Granularity-aligned chunks
- **Max Size**: Device VRAM (not fixed 32GB)
- **No reuse**: Requires reverse-order deallocation
- **Virtual Memory**: Uses CANN VMM API

#### Free Algorithm

```cpp
void free(void * ptr, size_t size) {
    pool_used -= size;
    // Assert: ptr must equal pool_addr + pool_used (reverse order)
    GGML_ASSERT(ptr == (void *)((char *)pool_addr + pool_used));
}
```

**Characteristics**:
- **Time Complexity**: O(1)
- **Constraint**: Must free in reverse order
- **Identical to CUDA**: Same stack discipline

---

## Algorithm Comparison Summary

### Selection Strategies

| Pool Type | Strategy | Time Complexity | Best For |
|-----------|----------|-----------------|----------|
| **CUDA Legacy** | Best-fit (min diff) | O(n) | Minimizing waste |
| **CANN Buffer** | First-fit (margin) | O(n) | Fast allocation |
| **CANN Priority** | Smallest-fit (heap) | O(log n) | Optimal reuse |
| **CUDA/CANN VMM** | Stack (LIFO) | O(1) | Sequential patterns |

### Reuse Logic Comparison

| Pool Type | Reuse Criteria | Margin | Cleanup |
|-----------|----------------|--------|---------|
| **CUDA Legacy** | Exact match preferred, then best-fit | None | None |
| **CANN Buffer** | First-fit with 4MB margin | 4MB | Time-based (>100ms) |
| **CANN Priority** | Smallest-fit with 4MB margin | 4MB | Time-based (>100ms) |
| **CUDA/CANN VMM** | No reuse (stack) | N/A | None |

### Memory Growth Strategies

| Pool Type | Growth Strategy | Lookahead |
|-----------|----------------|-----------|
| **CUDA Legacy** | 5% lookahead + 256-byte alignment | Yes |
| **CANN Buffer** | Exact size | No |
| **CANN Priority** | Exact size | No |
| **CUDA/CANN VMM** | Granularity-aligned chunks | No |

### Key Differences

1. **CUDA Legacy**:
   - Uses 5% lookahead to reduce allocations
   - No cleanup mechanism (buffers accumulate)
   - Best-fit selection minimizes waste

2. **CANN Buffer**:
   - Exact allocation (no lookahead)
   - Time-based cleanup prevents accumulation
   - First-fit with margin tolerance

3. **CANN Priority**:
   - Heap-based selection (O(log n))
   - Prefers smallest suitable buffer
   - Most efficient for large pools

4. **VMM Pools** (Both):
   - Stack-based (O(1) allocation)
   - No reuse (requires reverse-order free)
   - Best for sequential allocation patterns

### Performance Trade-offs

| Metric | CUDA Legacy | CANN Buffer | CANN Priority | VMM |
|--------|-------------|-------------|----------------|-----|
| **Allocation Speed** | Medium (O(n)) | Medium (O(n)) | Slow (O(log n)) | Fast (O(1)) |
| **Memory Efficiency** | High (best-fit) | Medium | High (smallest-fit) | High (no fragmentation) |
| **Fragmentation** | Medium | Medium | Low | None |
| **Cleanup Overhead** | None | Medium | Medium | None |
| **Scalability** | Limited (256 slots) | Limited (256 slots) | Good (heap) | Excellent |

### Use Case Recommendations

- **CUDA Legacy**: Good for predictable workloads with moderate buffer count
- **CANN Buffer**: Good for general-purpose workloads with cleanup needs
- **CANN Priority**: Best for large pools with many buffers (>100)
- **VMM**: Best for sequential allocation patterns (graph execution)

---

## Implementation Details

### CUDA Pool Selection

```c
std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device) {
    #if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return new ggml_cuda_pool_vmm(device);  // VMM if supported
    }
    #endif
    return new ggml_cuda_pool_leg(device);  // Legacy otherwise
}
```

### CANN Pool Selection

```cpp
std::unique_ptr<ggml_cann_pool> new_pool_for_device(int device) {
    std::string mem_pool_type = get_env("GGML_CANN_MEM_POOL").value_or("");
    
    if (mem_pool_type == "prio") {
        return new ggml_cann_pool_buf_prio(device);  // Explicit priority
    }
    
    if (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        return new ggml_cann_pool_vmm(device);  // VMM if supported
    }
    
    return new ggml_cann_pool_buf(device);  // Default buffer pool
}
```

**CANN allows explicit pool selection via `GGML_CANN_MEM_POOL` environment variable**, while CUDA automatically selects based on VMM support.

---

## Algorithmic Patterns Analysis

### Pattern 1: Linear Search with Best-Fit (CUDA Legacy)

**Algorithm**: Linear scan with minimum difference tracking

```c
best_diff = MAX_VALUE
for each buffer in pool:
    if buffer.size >= requested_size:
        diff = buffer.size - requested_size
        if diff < best_diff:
            best_diff = diff
            best_index = i
            if diff == 0: break  // Early exit optimization
```

**Complexity**: 
- Best case: O(1) - exact match found early
- Average case: O(n/2) - scan half the pool
- Worst case: O(n) - scan entire pool

**Optimizations**:
- Early exit on exact match
- Tracks minimum difference (best-fit)

**Trade-offs**:
- ✅ Minimizes memory waste
- ✅ Simple implementation
- ❌ O(n) complexity
- ❌ No cleanup mechanism

---

### Pattern 2: Linear Search with First-Fit and Cleanup (CANN Buffer)

**Algorithm**: Linear scan with time-based eviction

```cpp
for each buffer in pool:
    if buffer.used: continue
    
    if buffer.size >= requested_size:
        margin = buffer.size - requested_size
        if margin <= MAX_REUSE_MARGIN:
            return buffer  // First-fit with margin
    
    // Cleanup old buffers during scan
    if buffer.age > CLEANUP_THRESHOLD && buffer.size > MIN_SIZE:
        free(buffer)
        mark_as_empty(buffer)
```

**Complexity**:
- Best case: O(1) - first buffer fits
- Average case: O(n) - scan until match or cleanup
- Worst case: O(n) - full scan with cleanup

**Optimizations**:
- Combines search and cleanup in single pass
- Margin-based reuse (4MB tolerance)
- Time-based eviction prevents accumulation

**Trade-offs**:
- ✅ Prevents memory accumulation
- ✅ Margin tolerance improves hit rate
- ❌ O(n) complexity
- ❌ Cleanup overhead during allocation

---

### Pattern 3: Heap-Based Smallest-Fit (CANN Priority)

**Algorithm**: Min-heap extraction with cleanup

```cpp
candidates = []
while heap not empty:
    buffer = heap.pop()  // O(log n)
    
    if buffer.size >= requested_size:
        margin = buffer.size - requested_size
        if margin <= MAX_REUSE_MARGIN:
            return buffer  // Smallest suitable buffer
    
    if should_cleanup(buffer):
        free(buffer)
    else:
        candidates.push(buffer)

// Rebuild heap with remaining buffers
for buffer in candidates:
    heap.push(buffer)  // O(log n) each
```

**Complexity**:
- Best case: O(log n) - top element fits
- Average case: O(k log n) where k = buffers examined
- Worst case: O(n log n) - examine all buffers

**Optimizations**:
- Heap property ensures smallest-first selection
- Efficient for large pools (logarithmic operations)
- Cleanup integrated into extraction loop

**Trade-offs**:
- ✅ Optimal buffer selection (smallest-fit)
- ✅ Logarithmic complexity (better scaling)
- ❌ More complex implementation
- ❌ Overhead for small pools

---

### Pattern 4: Stack-Based Allocation (VMM Pools)

**Algorithm**: Stack pointer with virtual memory mapping

```c
if pool_used + size > pool_size:
    reserve_size = align(size - (pool_size - pool_used), granularity)
    
    if pool_addr == 0:
        reserve_virtual_address_space(MAX_SIZE)
    
    allocate_physical_memory(reserve_size)
    map_to_virtual_address(pool_addr + pool_size, reserve_size)
    pool_size += reserve_size

ptr = pool_addr + pool_used
pool_used += size
return ptr
```

**Complexity**:
- Allocation: O(1) - stack pointer increment
- Expansion: O(1) - virtual memory mapping (amortized)
- Free: O(1) - stack pointer decrement

**Optimizations**:
- Zero fragmentation (stack discipline)
- Virtual memory allows large address space
- Granularity-aligned expansion reduces syscalls

**Trade-offs**:
- ✅ O(1) allocation/free
- ✅ No fragmentation
- ✅ Supports very large pools
- ❌ Requires reverse-order deallocation
- ❌ No reuse (allocation pattern must be stack-like)

---

## Complexity Analysis

### Time Complexity Summary

| Operation | CUDA Legacy | CANN Buffer | CANN Priority | VMM |
|-----------|-------------|-------------|---------------|-----|
| **Allocate** | O(n) | O(n) | O(k log n) | O(1) |
| **Free** | O(n) | O(n) | O(log n) | O(1) |
| **Cleanup** | N/A | O(n) | O(k log n) | N/A |
| **Search** | O(n) | O(n) | O(log n) | N/A |

Where:
- `n` = number of buffers in pool (max 256 for fixed arrays)
- `k` = number of buffers examined during allocation (≤ n)

### Space Complexity

| Pool Type | Space Overhead | Notes |
|-----------|----------------|-------|
| **CUDA Legacy** | O(n) | Fixed array of 256 buffers |
| **CANN Buffer** | O(n) | Fixed array of 256 buffers |
| **CANN Priority** | O(n) | Heap + HashMap (2n) |
| **VMM** | O(1) | Only stack pointers |

---

## Memory Efficiency Analysis

### Fragmentation Comparison

1. **CUDA Legacy**:
   - **Internal fragmentation**: Low (best-fit minimizes waste)
   - **External fragmentation**: Medium (buffers scattered in array)
   - **Waste**: ~5% lookahead overhead

2. **CANN Buffer**:
   - **Internal fragmentation**: Medium (4MB margin tolerance)
   - **External fragmentation**: Medium (first-fit can leave gaps)
   - **Waste**: Up to 4MB per buffer (margin tolerance)

3. **CANN Priority**:
   - **Internal fragmentation**: Low (smallest-fit minimizes waste)
   - **External fragmentation**: Low (heap maintains order)
   - **Waste**: Up to 4MB per buffer (margin tolerance)

4. **VMM**:
   - **Internal fragmentation**: None (exact allocation)
   - **External fragmentation**: None (stack discipline)
   - **Waste**: None (perfect packing)

### Memory Growth Patterns

| Pool Type | Initial | Growth | Max Size |
|-----------|---------|--------|----------|
| **CUDA Legacy** | 0 | 5% lookahead per alloc | Unlimited (256 slots) |
| **CANN Buffer** | 0 | Exact size | Unlimited (256 slots) |
| **CANN Priority** | 0 | Exact size | Unlimited (heap) |
| **CUDA VMM** | 0 | Granularity chunks | 32 GB |
| **CANN VMM** | 0 | Granularity chunks | Device VRAM |

---

## Practical Recommendations

### When to Use Each Pool Type

1. **CUDA Legacy** (`ggml_cuda_pool_leg`):
   - ✅ Moderate number of buffers (< 100)
   - ✅ Predictable allocation patterns
   - ✅ Memory efficiency is priority
   - ❌ Avoid if buffer count > 200

2. **CANN Buffer** (`ggml_cann_pool_buf`):
   - ✅ General-purpose workloads
   - ✅ Need cleanup mechanism
   - ✅ Moderate buffer count
   - ❌ Avoid if many small buffers (cleanup overhead)

3. **CANN Priority** (`ggml_cann_pool_buf_prio`):
   - ✅ Large number of buffers (> 100)
   - ✅ Variable allocation sizes
   - ✅ Optimal memory efficiency needed
   - ❌ Overhead not worth it for < 50 buffers

4. **VMM Pools** (Both):
   - ✅ Sequential allocation patterns
   - ✅ Large memory requirements
   - ✅ Graph execution (reverse-order free)
   - ❌ Avoid if random allocation order

### Performance Characteristics

**Fastest Allocation**: VMM (O(1))
**Most Memory Efficient**: VMM (no fragmentation) or Priority (smallest-fit)
**Best Scalability**: Priority (O(log n)) or VMM (O(1))
**Best for Cleanup**: CANN Buffer/Priority (time-based eviction)
**Best for Reuse**: Priority (heap-based selection)

---

## Conclusion

The pooling algorithms represent different trade-offs:

- **CUDA Legacy**: Simple best-fit, good for moderate workloads
- **CANN Buffer**: First-fit with cleanup, good general-purpose solution
- **CANN Priority**: Heap-based smallest-fit, best for large pools
- **VMM**: Stack-based, best for sequential patterns

CANN provides more flexibility with explicit pool selection and cleanup mechanisms, while CUDA focuses on simplicity with automatic selection. Both VMM implementations are nearly identical, leveraging their respective virtual memory APIs for efficient large-scale allocation.
