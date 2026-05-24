#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H

#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <base/base.h>
namespace kernel {
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   const tensor::Tensor& block_table_tensor, int32_t block_size,
                   base::DeviceType device_type, CudaConfig* config);

// --- 新增：为 Continuous Batching 打造的 Batched MHA 接口 ---
void batched_mha_kernel_cu(int32_t batch_size, const tensor::Tensor& pos_tensor,
                           int32_t head_num, int32_t layer_index, int32_t seq_len,
                           int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                           const tensor::Tensor& mha_out,
                           const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                           const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                           const tensor::Tensor& block_table_tensor, int32_t block_size,
                           int32_t max_blocks_per_req, int32_t total_blocks_pool,
                           base::DeviceType device_type, CudaConfig* config);
}
#endif  // MHA_KERNEL_H
