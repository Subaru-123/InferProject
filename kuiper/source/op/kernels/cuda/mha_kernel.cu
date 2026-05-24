#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}


__global__ void paged_multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache_pool,
                                            float* value_cache_pool, int32_t* block_table, int32_t block_size,
                                            int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset_pool,
                                            int32_t total_blocks_pool) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // head当前的注意力头索引，kv_mul用于gqa，head_size表示一个自注意力头的维度
  // kv_dim = head_size * head_num，多头自注意力情况下的key,value 维度
  // kv_dim = head_size * head_num / kv_num，GQA情况下的key,value 维度
  int head_offset = (head / kv_mul) * head_size;
  // 计算自注意力分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    // A.计算逻辑块与块内偏移
    int logical_block_idx = t / block_size;
    int token_offset = t % block_size;

    // B.查表获取物理块ID
    int physical_block_idx = block_table[logical_block_idx];
    if (physical_block_idx < 0 || physical_block_idx >= total_blocks_pool) continue;

    // C.计算在物理显存池中的真实指针偏移
    // 偏移逻辑：层偏移 + 物理块偏移 + 块内Token偏移 + 头偏移
    // 显存池形状为 [num_layers, num_blocks, block_size, kv_dim]
    int64_t physical_offset = layer_offset_pool + physical_block_idx * block_size * kv_dim +
                              token_offset * kv_dim + head_offset;

    float* key_head = key_cache_pool + physical_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      // 1. Paged Attention 重新计算V的物理偏移
      int logical_block_idx = t / block_size;
      int token_offset = t % block_size;
      int physical_block_idx = block_table[logical_block_idx];
    if (physical_block_idx < 0 || physical_block_idx >= total_blocks_pool) continue;
      int64_t physical_offset = layer_offset_pool + physical_block_idx * block_size * kv_dim + token_offset * kv_dim + head_offset;
      float* value_head = value_cache_pool + physical_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   const tensor::Tensor& block_table_tensor, int32_t block_size,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  int32_t* block_table = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());

  cudaStream_t stream = config->stream;
  int32_t total_blocks_pool = (seq_len + block_size - 1) / block_size;
  paged_multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, block_table, block_size, kv_dim, kv_mul,
      head_num, head_size, layer_offset, total_blocks_pool);
}

__global__ void batched_paged_multi_head_attention_kernel(
    const int32_t* pos_array, int32_t seq_len, float* query, float* score_ptr, float* output, 
    float* key_cache_pool, float* value_cache_pool, const int32_t* block_table_2d, 
    int32_t block_size, int32_t max_blocks_per_req, int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size, int32_t layer_idx, int32_t total_blocks_pool) {
  
  int head = blockIdx.x;
  int batch_idx = blockIdx.y; // 新增：当前处理的是 Batch 中的哪一个请求
  if (head >= head_num) return;

  // 获取当前请求独占的参数
  int32_t pos = pos_array[batch_idx];
  const int32_t* block_table = block_table_2d + batch_idx * max_blocks_per_req;

  // 1024 物理块的严格偏移约束
  int64_t layer_offset_pool = (int64_t)layer_idx * total_blocks_pool * block_size * kv_dim;

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  
  // Q, Score, Output 指针按 Batch 偏移
  float* query_head = query + batch_idx * (head_num * head_size) + head * head_size;
  float* score_head = score_ptr + batch_idx * (head_num * seq_len) + head * seq_len;
  float* output_head = output + batch_idx * (head_num * head_size) + head * head_size;

  // 预加载 query 到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  int head_offset = (head / kv_mul) * head_size;

  // 计算自注意力分数 (Q @ K^T)
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    int logical_block_idx = t / block_size;
    int token_offset = t % block_size;
    int physical_block_idx = block_table[logical_block_idx];
    if (physical_block_idx < 0 || physical_block_idx >= total_blocks_pool) continue;

    int64_t physical_offset = layer_offset_pool + 
                              (int64_t)physical_block_idx * block_size * kv_dim +
                              (int64_t)token_offset * kv_dim + head_offset;

    float* key_head = key_cache_pool + physical_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y + 
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  // GPU Softmax，因为每一个 Block 都是完全隔离的
  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // 使用分数对 V 进行加权 (Score @ V)
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      int logical_block_idx = t / block_size;
      int token_offset = t % block_size;
      int physical_block_idx = block_table[logical_block_idx];
    if (physical_block_idx < 0 || physical_block_idx >= total_blocks_pool) continue;
      
      int64_t physical_offset = layer_offset_pool + 
                                (int64_t)physical_block_idx * block_size * kv_dim + 
                                (int64_t)token_offset * kv_dim + head_offset;
      
      float* value_head = value_cache_pool + physical_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

// 供上层调用的 Batched 启动接口
void batched_mha_kernel_cu(int32_t batch_size, const tensor::Tensor& pos_tensor,
                           int32_t head_num, int32_t layer_index, int32_t seq_len,
                           int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                           const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                           const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                           const tensor::Tensor& block_table_tensor, int32_t block_size,
                           int32_t max_blocks_per_req, int32_t total_blocks_pool,
                           base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  int32_t* block_table = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());
  int32_t* pos_array = const_cast<int32_t*>(pos_tensor.ptr<int32_t>());

  cudaStream_t stream = config->stream;
  
  // --- 核心升维：Grid 的 Y 轴被设置为 batch_size ---
  dim3 grid(head_num, batch_size);
  dim3 block(thread_num);

  batched_paged_multi_head_attention_kernel<<<grid, block, head_size * sizeof(float), stream>>>(
      pos_array, seq_len, query, score, output, key_cache, value_cache, block_table, 
      block_size, max_blocks_per_req, kv_dim, kv_mul, head_num, head_size, 
      layer_index, total_blocks_pool);
}

}  // namespace kernel