# All Pool Types in the Codebase

This document lists all memory pool implementations available across different backends.

## Summary

| Backend | Pool Types | Total |
|---------|-----------|-------|
| **CUDA** | Legacy, VMM | 2 |
| **CANN** | Buffer, Priority, VMM | 3 |
| **SYCL** | Legacy, Host | 2 |
| **Total** | | **7** |

---

## 1. CUDA Backend Pools

### 1.1 `ggml_cuda_pool_leg` (Legacy Pool)

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:317-411`

**Type**: Legacy buffer pool with best-fit selection

**Characteristics**:
- Fixed array of 256 buffers (`MAX_BUFFERS = 256`)
- Best-fit allocation algorithm (minimizes waste)
- 5% lookahead allocation (reduces future allocations)
- No cleanup mechanism
- O(n) allocation complexity

**Selection**: Automatic fallback when VMM is not supported

**Code**:
```c
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;
    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    // ... best-fit allocation logic
}
```

---

### 1.2 `ggml_cuda_pool_vmm` (Virtual Memory Pool)

**Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:415-521`

**Type**: Virtual memory pool with stack-based allocation

**Characteristics**:
- Stack-based allocation (LIFO)
- Maximum size: 32 GB (`CUDA_POOL_VMM_MAX_SIZE = 1ull << 35`)
- Uses CUDA Virtual Memory Management API
- O(1) allocation complexity
- Requires reverse-order deallocation
- No fragmentation

**Selection**: Automatic when device supports VMM

**Code**:
```c
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    // ... stack-based allocation
}
```

**Selection Logic**:
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

---

## 2. CANN Backend Pools

### 2.1 `ggml_cann_pool_buf` (Buffer Pool)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:398-592`

**Type**: Buffer pool with first-fit selection and cleanup

**Characteristics**:
- Fixed array of 256 buffers (`MAX_BUFFERS = 256`)
- First-fit allocation with 4MB margin tolerance
- Time-based cleanup (>100ms unused, >1MB)
- Exact-size allocation (no lookahead)
- O(n) allocation complexity

**Selection**: Default pool (when VMM not available or `GGML_CANN_MEM_POOL="leg"`)

**Code**:
```cpp
struct ggml_cann_pool_buf : public ggml_cann_pool {
    static const int MAX_BUFFERS = 256;
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB
    static const size_t min_free_margin = 1ull << 20;   // 1MB
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};
    // ... first-fit with cleanup
}
```

---

### 2.2 `ggml_cann_pool_buf_prio` (Priority Queue Pool)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:208-391`

**Type**: Priority queue-based pool with smallest-fit selection

**Characteristics**:
- Min-heap for buffer selection (smallest-fit)
- HashMap for O(1) buffer lookup
- 4MB margin tolerance for reuse
- Time-based cleanup (>100ms unused, >1MB)
- O(log n) allocation complexity
- Best for large pools

**Selection**: Explicit via `GGML_CANN_MEM_POOL="prio"`

**Code**:
```cpp
struct ggml_cann_pool_buf_prio : public ggml_cann_pool {
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB
    std::unordered_map<void *, size_t> buffer_pool;
    std::priority_queue<ggml_cann_buffer, 
                        std::vector<ggml_cann_buffer>, 
                        std::greater<>> free_buffers;
    // ... heap-based smallest-fit
}
```

---

### 2.3 `ggml_cann_pool_vmm` (Virtual Memory Pool)

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:600-756`

**Type**: Virtual memory pool with stack-based allocation

**Characteristics**:
- Stack-based allocation (LIFO)
- Maximum size: Device VRAM (not fixed)
- Uses CANN Virtual Memory Management API
- O(1) allocation complexity
- Requires reverse-order deallocation
- No fragmentation

**Selection**: Automatic when device supports VMM (unless `GGML_CANN_MEM_POOL="leg"`)

**Code**:
```cpp
struct ggml_cann_pool_vmm : public ggml_cann_pool {
    size_t max_size;  // Device VRAM
    void * pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    std::vector<aclrtDrvMemHandle> handles;
    // ... stack-based allocation
}
```

**Selection Logic**:
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

---

## 3. SYCL Backend Pools

### 3.1 `ggml_sycl_pool_leg` (Legacy Pool)

**Location**: `ggml/src/ggml-sycl/ggml-sycl.cpp:1198-1298`

**Type**: Legacy buffer pool with best-fit selection

**Characteristics**:
- Fixed array of 256 buffers (`MAX_SYCL_BUFFERS = 256`)
- Best-fit allocation algorithm (identical to CUDA legacy)
- 5% lookahead allocation
- No cleanup mechanism
- O(n) allocation complexity
- Uses SYCL device memory (`sycl::malloc_device`)

**Selection**: Default for device pools

**Code**:
```cpp
struct ggml_sycl_pool_leg : public ggml_sycl_pool {
    static const int MAX_SYCL_BUFFERS = 256;
    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    // ... best-fit allocation (similar to CUDA)
}
```

---

### 3.2 `ggml_sycl_pool_host` (Host Pool)

**Location**: `ggml/src/ggml-sycl/ggml-sycl.cpp:1300-1372`

**Type**: Host memory pool with round-robin allocation

**Characteristics**:
- Fixed array of 64 buffers (`MAX_POOL_SIZE = 64`)
- Round-robin allocation (circular buffer)
- Uses SYCL host memory (`sycl::malloc_host`)
- Simpler allocation logic
- O(1) allocation complexity
- Separate pool for host memory

**Selection**: Used for host memory allocations

**Code**:
```cpp
struct ggml_sycl_pool_host : public ggml_sycl_pool {
    static constexpr int MAX_POOL_SIZE = 64;
    std::vector<ggml_sycl_buffer> buffer_pool;
    inline static int counter{ 0 };  // Round-robin counter
    // ... round-robin allocation
}
```

**Selection Logic**:
```cpp
std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device) {
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device) {
    // VMM not yet implemented for SYCL
    return std::unique_ptr<ggml_sycl_pool>(new ggml_sycl_pool_leg(qptr, device));
}
```

**Note**: SYCL VMM pool is commented out (not yet implemented):
```cpp
// TBD pool with virtual memory management
// struct ggml_sycl_pool_vmm : public ggml_sycl_pool
```

---

## Pool Type Comparison Table

| Pool Type | Backend | Selection Strategy | Cleanup | Complexity | Max Size | Special Features |
|-----------|---------|-------------------|---------|------------|----------|------------------|
| `ggml_cuda_pool_leg` | CUDA | Best-fit | None | O(n) | Unlimited (256 slots) | 5% lookahead |
| `ggml_cuda_pool_vmm` | CUDA | Stack (LIFO) | None | O(1) | 32 GB | Virtual memory |
| `ggml_cann_pool_buf` | CANN | First-fit | Time-based | O(n) | Unlimited (256 slots) | 4MB margin, cleanup |
| `ggml_cann_pool_buf_prio` | CANN | Smallest-fit (heap) | Time-based | O(log n) | Unlimited | Heap-based, optimal |
| `ggml_cann_pool_vmm` | CANN | Stack (LIFO) | None | O(1) | Device VRAM | Virtual memory |
| `ggml_sycl_pool_leg` | SYCL | Best-fit | None | O(n) | Unlimited (256 slots) | 5% lookahead |
| `ggml_sycl_pool_host` | SYCL | Round-robin | None | O(1) | 64 buffers | Host memory only |

---

## Selection Mechanisms

### CUDA
- **Automatic**: VMM if supported, otherwise Legacy
- **No user control**: Selection is automatic

### CANN
- **Configurable**: Via `GGML_CANN_MEM_POOL` environment variable
  - `"prio"` → Priority queue pool
  - `"leg"` → Buffer pool (legacy)
  - `""` (default) → VMM if supported, else Buffer pool

### SYCL
- **Fixed**: Legacy for device, Host for host memory
- **No VMM**: VMM pool not yet implemented

---

## Base Classes

All pools inherit from backend-specific base classes:

- **CUDA**: `ggml_cuda_pool` (abstract base)
- **CANN**: `ggml_cann_pool` (abstract base)
- **SYCL**: `ggml_sycl_pool` (abstract base)

Each base class defines:
- `virtual void * alloc(size_t size, size_t * actual_size) = 0;`
- `virtual void free(void * ptr, size_t size) = 0;`

---

## Usage Patterns

### CUDA
```c
ggml_cuda_pool & pool = ctx.pool(device);
ggml_cuda_pool_alloc<float> temp_buffer(pool, size);
```

### CANN
```cpp
ggml_cann_pool & pool = ctx.pool();
ggml_cann_pool_alloc<float> temp_buffer(pool, size);
```

### SYCL
```cpp
ggml_sycl_pool & pool = ctx.pool(device);
ggml_sycl_pool_alloc<float> temp_buffer(pool, size);
```

---

## Summary

**Total Pool Types**: 7
- **CUDA**: 2 pools (Legacy, VMM)
- **CANN**: 3 pools (Buffer, Priority, VMM)
- **SYCL**: 2 pools (Legacy, Host)

**Common Patterns**:
- Legacy pools: Best-fit or first-fit with fixed arrays
- VMM pools: Stack-based with virtual memory (CUDA & CANN)
- Special pools: Priority queue (CANN), Host memory (SYCL)

**Selection**:
- CUDA: Automatic
- CANN: Configurable via environment variable
- SYCL: Fixed (device vs host)
