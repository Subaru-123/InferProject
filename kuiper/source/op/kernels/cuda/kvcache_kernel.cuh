#ifndef KUIPER_OP_KERNELS_CUDA_KVCACHE_KERNEL_CUH
#define KUIPER_OP_KERNELS_CUDA_KVCACHE_KERNEL_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace kernel {

// C++ 宿主接口：启动 Batched Paged KV 写入核函数
void batched_write_paged_kv_cu(
    const float* k_src, 
    const float* v_src, 
    const int32_t* pos_array, 
    const int32_t* block_table_2d, 
    float* key_cache_pool, 
    float* value_cache_pool, 
    int32_t batch_size, 
    int32_t kv_dim, 
    int32_t block_size, 
    int32_t max_blocks_per_req, 
    int32_t layer_idx, 
    int32_t total_blocks,
    cudaStream_t stream);

} // namespace kernel

#endif // KUIPER_OP_KERNELS_CUDA_KVCACHE_KERNEL_CUH