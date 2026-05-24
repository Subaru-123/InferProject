#include "rope_kernel.cuh"
namespace kernel {

#if defined (LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;
  *vec_ptr =
      make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float fci = *(sin_cache + pos * head_size + head_dim);
  float fcr = *(cos_cache + pos * head_size + head_dim);

  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()));
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()));
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>(), sin_cache.ptr<float>(),
                                             cos_cache.ptr<float>());
  }
}

// -------------------------------------------------------------
// 【修复后的 Batched RoPE Kernel】
// 支持 Continuous Batching，并适配交错格式的模型权重 (相邻两元素旋转)
// -------------------------------------------------------------
__global__ void batched_rope_kernel(
    int32_t batch_size,
    int32_t q_dim,
    int32_t kv_dim,
    int32_t head_size,
    const int32_t* pos_array,
    const float* sin_cache,
    const float* cos_cache,
    float* q_array,
    float* k_array) 
{
    // Y 轴：对应 Batch 中的第几个并发请求
    int batch_idx = blockIdx.y; 
    
    // X 轴：每个线程负责处理一对（即两个相邻的 float），所以索引是总长度的一半
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 防越界保护
    if (batch_idx >= batch_size) return;
 
    // 当前线程处理的一对浮点数在单条特征向量（dim）中的起始偏移位置
    int idx = dim_idx * 2;
    if (idx >= q_dim) return;
 
    // 查表：当前请求对应的实际上下文位置（pos）
    int32_t pos = pos_array[batch_idx];
 
    // 计算在此特征头内部的偏移，以定位该取哪个三角函数值
    int head_dim = idx % head_size;
    
    // 从缓存表中读取正确的旋转频率
    // 注意：这里的读取步长与老版本完全一致
    float fci = *(sin_cache + pos * head_size + head_dim);
    float fcr = *(cos_cache + pos * head_size + head_dim);
 
    // 计算当前并发请求在 Q 矩阵中的全局一维偏移
    int q_offset = batch_idx * q_dim + idx;
    
    // 复用 device 函数执行相邻浮点数的 2D 旋转
    rope_calc(fcr, fci, q_array, q_offset);
 
    // K 矩阵维度一般小于等于 Q (支持 GQA)，确保当前偏移在 K 的界限内再做计算
    if (idx < kv_dim) {
        int k_offset = batch_idx * kv_dim + idx;
        rope_calc(fcr, fci, k_array, k_offset);
    }
}

// 供 C++ 调用的 Launch 函数
void batched_rope_kernel_cu(
    int32_t batch_size,
    int32_t q_dim,
    int32_t kv_dim,
    int32_t head_size,
    const int32_t* pos_array,
    const float* sin_cache,
    const float* cos_cache,
    float* q_array,
    float* k_array,
    cudaStream_t stream) 
{
    // 每个线程处理一对数据，因此所需的最大线程跨度是 max(q_dim, kv_dim) / 2
    int32_t max_dim_half = (q_dim > kv_dim ? q_dim : kv_dim) / 2;
    
    dim3 block(128);
    // Y 轴的 Grid 大小直接设置为 batch_size，完美契合 Batched 处理
    dim3 grid((max_dim_half + block.x - 1) / block.x, batch_size);

    batched_rope_kernel<<<grid, block, 0, stream>>>(
        batch_size, q_dim, kv_dim, head_size, pos_array, 
        sin_cache, cos_cache, q_array, k_array);
}
}  // namespace kernel