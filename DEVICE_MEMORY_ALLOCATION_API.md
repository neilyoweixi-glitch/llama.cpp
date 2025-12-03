# Device Memory Allocation API

This document describes how GPU and NPU memory is allocated in the codebase, tracing the API calls from the public interface down to the low-level allocation functions (`cudaMalloc`, `cudaMallocManaged`, and `aclrtMalloc`).

## Overview

The codebase uses a backend abstraction layer for device memory allocation. The main entry points are:

- **CUDA/GPU**: `ggml_backend_buft_alloc_buffer()` → `ggml_backend_cuda_buffer_type_alloc_buffer()` → `ggml_cuda_device_malloc()` → `cudaMalloc()` / `cudaMallocManaged()`
- **CANN/NPU**: `ggml_backend_buft_alloc_buffer()` → `ggml_backend_cann_buffer_type_alloc_buffer()` → `aclrtMalloc()`

## Public API Entry Points

### CUDA (GPU) Memory Allocation

**Header**: `ggml/include/ggml-cuda.h`

```c
// Get buffer type for a specific device
ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// Generic backend buffer allocation API (from ggml-backend.h)
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
```

**Usage Example**:
```c
ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);  // device 0
ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
```

### CANN (NPU) Memory Allocation

**Header**: `ggml/include/ggml-cann.h`

```c
// Get buffer type for a specific device
ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device);

// Generic backend buffer allocation API (from ggml-backend.h)
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
```

**Usage Example**:
```c
ggml_backend_buffer_type_t buft = ggml_backend_cann_buffer_type(0);  // device 0
ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
```

## Call Chain Details

### CUDA (GPU) Allocation Path

1. **Public API**: `ggml_backend_buft_alloc_buffer(buft, size)`
   - **Location**: `ggml/include/ggml-backend.h:38`
   - **Implementation**: `ggml/src/ggml-backend.cpp`

2. **CUDA Buffer Type Allocator**: `ggml_backend_cuda_buffer_type_alloc_buffer()`
   - **Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:693-710`
   - **Function**:
   ```c
   static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(
       ggml_backend_buffer_type_t buft, size_t size)
   ```
   - Sets the CUDA device
   - Calls `ggml_cuda_device_malloc()` to allocate memory
   - Creates and initializes the buffer context

3. **CUDA Device Malloc**: `ggml_cuda_device_malloc()`
   - **Location**: `ggml/src/ggml-cuda/ggml-cuda.cu:111-136`
   - **Function**:
   ```c
   static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device)
   ```
   - **Logic**:
     - If `GGML_CUDA_ENABLE_UNIFIED_MEMORY` environment variable is set:
       - Calls `cudaMallocManaged()` (line 115)
       - For HIP, may fall back to `cudaMalloc()` if unified memory is not supported (line 129)
     - Otherwise:
       - Calls `cudaMalloc()` (line 133)

4. **Low-level CUDA Functions**:
   - `cudaMalloc()` - Standard device memory allocation
   - `cudaMallocManaged()` - Unified memory allocation (accessible from both host and device)
   - **Location**: Called via CUDA runtime API

### CANN (NPU) Allocation Path

1. **Public API**: `ggml_backend_buft_alloc_buffer(buft, size)`
   - **Location**: `ggml/include/ggml-backend.h:38`
   - **Implementation**: `ggml/src/ggml-backend.cpp`

2. **CANN Buffer Type Allocator**: `ggml_backend_cann_buffer_type_alloc_buffer()`
   - **Location**: `ggml/src/ggml-cann/ggml-cann.cpp:1384-1405`
   - **Function**:
   ```c
   static ggml_backend_buffer_t ggml_backend_cann_buffer_type_alloc_buffer(
       ggml_backend_buffer_type_t buft, size_t size)
   ```
   - Sets the CANN device
   - Pads size to 128-byte alignment
   - Calls `aclrtMalloc()` directly (line 1395)
   - Creates and initializes the buffer context

3. **Low-level CANN Function**:
   - `aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST)` - NPU device memory allocation
   - **Location**: `ggml/src/ggml-cann/ggml-cann.cpp:1395`
   - Uses `ACL_MEM_MALLOC_HUGE_FIRST` flag for huge page allocation

### CANN Pool Allocator (Alternative Path)

CANN also has a pool-based allocator for temporary buffers:

**Location**: `ggml/src/ggml-cann/ggml-cann.cpp:354`

The pool allocator (`ggml_cann_pool_buf::alloc()`) also calls `aclrtMalloc()`:
```cpp
ggml_cann_set_device(device);
ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
```

This is used internally for temporary buffers during operations (see `ggml/src/ggml-cann/aclnn_ops.cpp`).

## Key Files

### CUDA Implementation
- **Header**: `ggml/include/ggml-cuda.h`
- **Implementation**: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml_cuda_device_malloc()`: Lines 111-136
  - `ggml_backend_cuda_buffer_type_alloc_buffer()`: Lines 693-710

### CANN Implementation
- **Header**: `ggml/include/ggml-cann.h`
- **Implementation**: `ggml/src/ggml-cann/ggml-cann.cpp`
  - `ggml_backend_cann_buffer_type_alloc_buffer()`: Lines 1384-1405
  - Pool allocator: Lines 300-367

### Backend Abstraction
- **Header**: `ggml/include/ggml-backend.h`
- **Implementation**: `ggml/src/ggml-backend.cpp`

## Implementation Code Snippets

### CUDA: Low-level Allocation Function

**File**: `ggml/src/ggml-cuda/ggml-cuda.cu:111-136`

```c
static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);  // ← Unified memory allocation
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            CUDA_CHECK(cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
        }
        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            err = cudaMalloc(ptr, size);  // ← Fallback to standard allocation
        }
#endif
    } else {
        err = cudaMalloc(ptr, size);  // ← Standard device memory allocation
    }
    return err;
}
```

### CUDA: Buffer Type Allocator

**File**: `ggml/src/ggml-cuda/ggml-cuda.cu:693-710`

```c
static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = 
        (ggml_backend_cuda_buffer_type_context *)buft->context;
    
    ggml_cuda_set_device(buft_ctx->device);
    
    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", 
            __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }
    
    ggml_backend_cuda_buffer_context * ctx = 
        new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
    
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}
```

### CANN: Buffer Type Allocator

**File**: `ggml/src/ggml-cann/ggml-cann.cpp:1384-1405`

```cpp
static ggml_backend_buffer_t ggml_backend_cann_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cann_buffer_type_context * buft_ctx = 
        (ggml_backend_cann_buffer_type_context *) buft->context;
    
    ggml_cann_set_device(buft_ctx->device);
    
    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }
    
    void * dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);  // ← NPU allocation
    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", 
            __func__, size / 1024.0 / 1024.0, buft_ctx->device, aclGetRecentErrMsg());
        return nullptr;
    }
    
    ggml_backend_cann_buffer_context * ctx = 
        new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);
    
    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface, ctx, size);
}
```

### Backend Abstraction Layer

**File**: `ggml/src/ggml-backend.cpp:38-46`

```c
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    
    GGML_ASSERT(buft);
    return buft->iface.alloc_buffer(buft, size);  // ← Calls backend-specific allocator
}
```

## Summary

The device memory allocation API follows this pattern:

1. **Get buffer type**: `ggml_backend_cuda_buffer_type(device)` or `ggml_backend_cann_buffer_type(device)`
2. **Allocate buffer**: `ggml_backend_buft_alloc_buffer(buft, size)`
3. **Internal allocation**:
   - **CUDA**: `ggml_cuda_device_malloc()` → `cudaMalloc()` or `cudaMallocManaged()`
   - **CANN**: Direct call to `aclrtMalloc()`

The unified memory feature for CUDA can be enabled by setting the `GGML_CUDA_ENABLE_UNIFIED_MEMORY` environment variable, which will use `cudaMallocManaged()` instead of `cudaMalloc()`.

## Quick Reference

| Backend | Public API | Low-level Function | Location |
|---------|-----------|-------------------|----------|
| CUDA | `ggml_backend_buft_alloc_buffer()` | `cudaMalloc()` / `cudaMallocManaged()` | `ggml/src/ggml-cuda/ggml-cuda.cu:111-136` |
| CANN | `ggml_backend_buft_alloc_buffer()` | `aclrtMalloc()` | `ggml/src/ggml-cann/ggml-cann.cpp:1395` |
