# Pool Allocator with VMM Feature - Analysis

## Overview

The codebase implements a CUDA memory pool allocator that uses CUDA's Virtual Memory Management (VMM) feature for efficient memory allocation and deallocation. The VMM pool allocator (`ggml_cuda_pool_vmm`) provides a high-performance memory management system for GPU operations.

## Architecture

### Key Components

1. **`ggml_cuda_pool_vmm`** (`ggml/src/ggml-cuda/ggml-cuda.cu:415-521`)
   - Inherits from `ggml_cuda_pool` base class
   - Uses CUDA VMM APIs: `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`
   - Maximum pool size: 32 GB (`CUDA_POOL_VMM_MAX_SIZE = 1ull << 35`)
   - Manages a contiguous virtual address space with on-demand physical memory mapping

2. **`ggml_cuda_pool_alloc<T>`** (`ggml/src/ggml-cuda/common.cuh:887-928`)
   - RAII wrapper for pool allocations
   - Automatically frees memory when destructed
   - Used extensively throughout CUDA operations

3. **`ggml_backend_cuda_context`** (`ggml/src/ggml-cuda/common.cuh:975-1026`)
   - Manages pools per device: `pools[GGML_CUDA_MAX_DEVICES]`
   - Provides access via `pool(int device)` method

## How VMM Pool Allocator Works

### Allocation Process (`ggml_cuda_pool_vmm::alloc`)

1. **Size Alignment**: Rounds up to 128-byte alignment
2. **Check Available Space**: Compares requested size with `pool_size - pool_used`
3. **Expand Pool if Needed**:
   - Calculates `reserve_size` rounded to `granularity`
   - Creates physical memory allocation via `cuMemCreate`
   - Reserves virtual address space via `cuMemAddressReserve` (first time only)
   - Maps physical memory to virtual address via `cuMemMap`
   - Sets access permissions via `cuMemSetAccess`
   - Releases allocation handle (no longer needed after mapping)
4. **Return Pointer**: Returns pointer at `pool_addr + pool_used`, increments `pool_used`

### Deallocation Process (`ggml_cuda_pool_vmm::free`)

- **Stack-based deallocation**: Must be in reverse order of allocations
- Simply decrements `pool_used` counter
- Physical memory remains mapped until pool destruction
- On destruction: unmaps all memory and frees virtual address space

### Key VMM APIs Used

- `cuMemAddressReserve`: Reserves contiguous virtual address space
- `cuMemCreate`: Creates physical memory allocation
- `cuMemMap`: Maps physical memory to virtual address
- `cuMemSetAccess`: Sets access permissions for mapped memory
- `cuMemUnmap`: Unmaps memory (on destruction)
- `cuMemAddressFree`: Frees reserved virtual address space

## High-Level Functions That Exercise the Pool Allocator

### Primary Entry Points

1. **`ggml_backend_cuda_graph_compute`** (`ggml/src/ggml-cuda/ggml-cuda.cu:3552`)
   - Main entry point for executing CUDA computation graphs
   - Calls `ggml_cuda_compute_forward` for each node
   - **Usage**: Called via `ggml_backend_graph_compute` API

2. **`ggml_cuda_compute_forward`** (`ggml/src/ggml-cuda/ggml-cuda.cu:2409`)
   - Dispatches operations to specific CUDA implementations
   - Each operation may allocate temporary buffers from the pool

### Operations That Use Pool Allocations

The following operations allocate temporary memory from the pool (examples):

#### Matrix Operations (Heavy Pool Usage)
- **`ggml_cuda_mul_mat`** / **`ggml_cuda_mul_mat_q`** / **`ggml_cuda_mul_mat_id`**
  - Matrix multiplication operations
  - Allocates temporary buffers for quantized matrices, indices, expert bounds
  - **Example allocations**:
    - `ggml_cuda_pool_alloc<char> src1_q8_1` - quantized matrix buffers
    - `ggml_cuda_pool_alloc<int32_t> ids_src1` - index arrays
    - `ggml_cuda_pool_alloc<int32_t> expert_bounds` - expert routing arrays

#### Attention Operations
- **`ggml_cuda_flash_attn_ext`** (`ggml/src/ggml-cuda/fattn-common.cuh:802-806`)
  - Flash attention implementation
  - Allocates: `K_f16`, `V_f16`, `KV_max`, `dst_tmp`, `dst_tmp_meta`

#### Sorting/Indexing Operations
- **`ggml_cuda_op_argsort`** (`ggml/src/ggml-cuda/argsort.cu:32-34`)
  - Allocates: `temp_indices_alloc`, `temp_keys_alloc`, `offsets_alloc`, `temp_storage_alloc`

#### Reduction Operations
- **`ggml_cuda_op_sum`** (`ggml/src/ggml-cuda/sum.cu:15`)
  - Allocates: `tmp_alloc` for temporary reduction buffers

- **`ggml_cuda_op_mean`** (`ggml/src/ggml-cuda/mean.cu:53`)
  - Allocates: `tmp_alloc` for mean computation

#### Type Conversion Operations
- **`ggml_cuda_mul_mat`** (in `ggml-cuda.cu:1252-1334`)
  - Allocates temporary buffers for type conversions:
    - `src1_as_bf16`, `dst_bf16` - bfloat16 conversions
    - `src0_as_f16`, `src1_as_f16`, `dst_f16` - half precision conversions
    - `src0_ddq_as_f32`, `src1_ddq_as_f32` - dequantization buffers

#### Multi-Expert Operations
- **`ggml_cuda_mul_mat_f`** (`ggml/src/ggml-cuda/mmf.cu:44-46`)
  - Allocates: `ids_src_compact_dev`, `ids_dst_compact_dev`, `expert_bounds_dev`

#### Loss Functions
- **`ggml_cuda_cross_entropy_loss`** (`ggml/src/ggml-cuda/cross-entropy-loss.cu:123`)
  - Allocates: `dst_tmp` for temporary loss computation

## Recommended High-Level Functions for Tracing

To exercise and trace CUDA memory allocation/deallocation behavior, use these high-level functions:

### 1. **Matrix Multiplication Operations** (Most Pool Activity)
```cpp
// High-level API usage:
ggml_tensor * result = ggml_mul_mat(ctx, weight_matrix, input_matrix);
ggml_backend_graph_compute(backend, gf);  // Triggers pool allocations
```

**Operations triggered**:
- `GGML_OP_MUL_MAT` → `ggml_cuda_mul_mat` → multiple pool allocations
- `GGML_OP_MUL_MAT_ID` → `ggml_cuda_mul_mat_id` → expert routing allocations

### 2. **Attention Operations** (Moderate Pool Activity)
```cpp
ggml_tensor * attn = ggml_flash_attn(ctx, q, k, v, ...);
ggml_backend_graph_compute(backend, gf);
```

**Operations triggered**:
- `GGML_OP_FLASH_ATTN` → `ggml_cuda_flash_attn_ext` → K/V buffers, metadata

### 3. **Sorting/Indexing Operations** (Moderate Pool Activity)
```cpp
ggml_tensor * sorted = ggml_argsort(ctx, input);
ggml_backend_graph_compute(backend, gf);
```

**Operations triggered**:
- `GGML_OP_ARGSORT` → `ggml_cuda_op_argsort` → temp indices, keys, offsets

### 4. **Reduction Operations** (Light Pool Activity)
```cpp
ggml_tensor * sum = ggml_sum(ctx, input);
ggml_tensor * mean = ggml_mean(ctx, input);
ggml_backend_graph_compute(backend, gf);
```

**Operations triggered**:
- `GGML_OP_SUM` → `ggml_cuda_op_sum` → temporary reduction buffers
- `GGML_OP_MEAN` → `ggml_cuda_op_mean` → temporary buffers

### 5. **Complex Operations** (Heavy Pool Activity)
```cpp
// Multi-expert model forward pass
ggml_tensor * output = ggml_mul_mat_id(ctx, experts, input, expert_ids);
ggml_backend_graph_compute(backend, gf);
```

**Operations triggered**:
- Multiple pool allocations for expert routing, type conversions, temporary buffers

## Tracing CUDA Memory Allocations

### Enable Debug Logging

The codebase includes debug macros for tracing allocations:

1. **Enable `DEBUG_CUDA_MALLOC`**:
   ```cpp
   #define DEBUG_CUDA_MALLOC
   ```
   Located in `ggml/src/ggml-cuda/ggml-cuda.cu:314`

2. **Debug Output**:
   - Allocation: `printf("cuda pool[%d]: allocated %llu bytes at %llx\n", ...)`
   - Deallocation: `printf("cuda pool[%d]: freed %llu bytes at %llx\n", ...)`

### Using CUDA Memory Tracing Tools

1. **Nsight Compute / Nsight Systems**:
   ```bash
   nsys profile --trace=cuda,nvtx ./your_program
   ```

2. **CUDA Memory Trace APIs**:
   - Use `cuMemGetAllocationGranularity` to check granularity
   - Monitor `cuMemCreate`, `cuMemMap`, `cuMemUnmap` calls
   - Track `pool_used`, `pool_size` counters

3. **Custom Tracing**:
   Add instrumentation to `ggml_cuda_pool_vmm::alloc` and `free`:
   ```cpp
   void * alloc(size_t size, size_t * actual_size) override {
       // ... existing code ...
       // Add tracing:
       trace_allocation(device, size, *actual_size, ptr, pool_used, pool_size);
       return ptr;
   }
   ```

### Key Metrics to Track

1. **Allocation Patterns**:
   - Size distribution of allocations
   - Frequency of allocations
   - Allocation/deallocation order (should be LIFO)

2. **Pool Growth**:
   - `pool_size` over time (physical memory mapped)
   - `pool_used` over time (virtual memory used)
   - Granularity-aligned growth increments

3. **Memory Efficiency**:
   - Ratio of `pool_used` to `pool_size`
   - Fragmentation (should be minimal with stack-based deallocation)
   - Peak memory usage

## Example Workflow for Tracing

1. **Initialize CUDA Backend**:
   ```cpp
   ggml_backend_t backend = ggml_backend_cuda_init(0);  // device 0
   ```

2. **Build Computation Graph**:
   ```cpp
   ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, k);
   ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
   ggml_tensor * c = ggml_mul_mat(ctx, a, b);
   ggml_cgraph * gf = ggml_build_forward(c);
   ```

3. **Execute with Tracing**:
   ```cpp
   // Enable debug output or use external profiler
   ggml_backend_graph_compute(backend, gf);
   ```

4. **Analyze Trace**:
   - Identify allocation patterns
   - Check for proper LIFO deallocation
   - Monitor pool growth and peak usage
   - Verify granularity alignment

## Important Notes

1. **LIFO Deallocation**: The VMM pool requires deallocations in reverse order of allocations. This is enforced by assertions in `free()`.

2. **Granularity**: Allocations are rounded up to the device's VMM granularity (typically 2MB), which can cause memory overhead.

3. **Pool Lifetime**: The pool persists for the lifetime of the `ggml_backend_cuda_context`. Memory is only freed when the context is destroyed.

4. **Multi-Device**: Each device has its own pool (`pools[device]`), accessed via `ctx.pool(device)`.

5. **Fallback**: If VMM is not supported, the system falls back to `ggml_cuda_pool_leg` (legacy pool allocator).

## Files to Examine

- **Core Implementation**: `ggml/src/ggml-cuda/ggml-cuda.cu:414-521`
- **Pool Interface**: `ggml/src/ggml-cuda/common.cuh:879-928`
- **Usage Examples**: 
  - `ggml/src/ggml-cuda/mmq.cu` - Matrix multiplication with pool allocations
  - `ggml/src/ggml-cuda/argsort.cu` - Sorting with pool allocations
  - `ggml/src/ggml-cuda/fattn-common.cuh` - Attention with pool allocations
