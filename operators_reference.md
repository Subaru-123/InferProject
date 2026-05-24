# KuiperLLama 算子参考手册

> 覆盖项目中全部 10 个推理算子的 CPU/CUDA/batched 实现，含逐行注释代码与文字执行流程。

---

## 目录

1. [Embedding（嵌入查找）](#1-embedding嵌入查找)
2. [RMSNorm（均方根归一化）](#2-rmsnorm均方根归一化)
3. [Matmul（矩阵乘法）](#3-matmul矩阵乘法)
4. [RoPE（旋转位置编码）](#4-rope旋转位置编码)
5. [KV Cache Write（Paged KV 缓存写入）](#5-kv-cache-writepaged-kv-缓存写入)
6. [Softmax（注意力归一化）](#6-softmax注意力归一化)
7. [Multi-Head Attention（多头注意力）](#7-multi-head-attention多头注意力)
8. [SwiGLU（门控激活）](#8-swiglu门控激活)
9. [VecAdd（残差连接）](#9-vecadd残差连接)
10. [Argmax Sampler（贪心采样）](#10-argmax-sampler贪心采样)

---

## 1. Embedding（嵌入查找）

**功能**：将 token ID 列表转换为对应的稠密向量（embedding 查表）。

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/emb_kernel.cpp

void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  const int32_t input_num = static_cast<int32_t>(input.size());   // 输入 token 数量
  const int32_t weight_dim = weight.get_dim(1);                    // 每个 token 的向量维度 (dim)

  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = *input.ptr<int32_t>(i);  // 取出第 i 个 token ID
    if (token > vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    }
    // 在 weight 矩阵的第 token 行找到对应 embedding
    float* dest_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
    float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
    // 将 [1, dim] 的 embedding 向量拷贝到输出
    allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float), kMemcpyCPU2CPU);
  }
}
```

### CUDA 实现（天然支持 batch）

```cpp
// 文件: kuiper/source/op/kernels/cuda/emb_kernel.cu

__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) {
  int32_t token_idx = blockIdx.x;               // 每个 block 处理一个 token
  if (token_idx >= token_num) return;

  int32_t token = input_ptr[token_idx];          // 查表
  if (token >= vocab_size) return;

  float* output_ptr_start = output_ptr + token_idx * weight_dim;
  const float* weight_ptr_start = weight_ptr + token * weight_dim;

  // 线程并发拷贝 embedding 向量的各分量
  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}
```

> 单请求和多请求使用**同一个 kernel**，区别仅在于 `token_num`（即 `grid.x` 的大小）。

### 执行流程

```
输入: [token_0, token_1, ..., token_{B-1}]    shape: [batch_size]
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      token_0 ──查表──▶ weight[token_0, :]  ──▶ output[0, :]
      token_1 ──查表──▶ weight[token_1, :]  ──▶ output[1, :]
      ...
      token_{B-1} ─查表─▶ weight[token_{B-1}, :] ──▶ output[B-1, :]

输出: [batch_size, dim]
```

---

## 2. RMSNorm（均方根归一化）

**功能**：对输入向量做 RMS 归一化 + 可学习权重缩放。公式：`out = weight * (input * rsqrt(mean(input²) + ε))`

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/rmsnorm_kernel.cpp

void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();
  const int32_t dim = static_cast<int32_t>(input.size());

  // 使用 Armadillo 库向量化计算
  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

  // Qwen 使用 eps=1e-6，LLaMA 使用 eps=1e-5
  const float eps = 1e-6f;

  // step 1: 计算均方值 mean(in²)
  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  // step 2: 计算逆平方根
  const float rsqrt = 1.f / std::sqrt(mean);
  // step 3: weight ⊙ (rsqrt * input)
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}
```

### CUDA 实现（单请求，一维）

```cpp
// 文件: kuiper/source/op/kernels/cuda/rmsnorm_kernel.cu

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;
  float sum = 0.0f;

  // 阶段 1: 用 float4 向量化加载，计算 sum(in²)
  float4* in_pack = reinterpret_cast<float4*>(in);
  const int pack_num = size / 4;
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = in_pack[i];
    sum += in_float4.x * in_float4.x + in_float4.y * in_float4.y +
           in_float4.z * in_float4.z + in_float4.w * in_float4.w;
  }
  // 处理尾部（不足 4 的部分）
  for (int i = pack_num * 4 + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  // 阶段 2: CUB BlockReduce 归约求和
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) shared_val = sum;
  __syncthreads();
  sum = shared_val;

  // 阶段 3: 计算 scale 并写回
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = in_pack[i];
    float4 wei_float4 = wei_pack[i];
    out_pack[i] = make_float4(scale * in_float4.x * wei_float4.x, ...);
  }
}
```

### CUDA 实现（多维 batched）

```cpp
// 同上文件, 不同入口函数

static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out,
                                           int dim_size, int size, float eps) {
  const int bid = blockIdx.x;       // 第几行（第几个 batch item）
  if (bid >= dim_size) return;

  float* block_in = in + bid * size;    // 定位到当前 batch item 的数据
  float* block_out = out + bid * size;

  // 后续计算与单请求版本完全相同，仅数据指针按行偏移
  // ...
}

// 调用: grid = dim_size (即 batch_size), block = 128
// row_rmsnorm_f32_dim<<<dim_size, 128>>>(
//     in_ptr, wei_ptr, out_ptr, dim_size, dim, eps);
```

> 注意：项目当前在 `attention_rms`、`feed_forward`、`cls_logits` 中**显式用 for 循环逐 batch item 调用**一维 RMSNorm，而非调用多维版本。源码注释标记为"核心修复：将大矩阵切片，彻底阻断 RMSNorm 跨批次求和带来的污染"。

### 执行流程

```
输入: x [dim]  或  [batch_size, dim]
                           │
          ┌────────────────┼────────────────┐
          ▼                                  ▼
   计算 mean(x²) + ε               weight [dim] (可学习参数)
          │                                  │
          ▼                                  │
   rsqrt = 1 / sqrt(mean)                     │
          │                                  │
          ▼                                  │
   tmp = rsqrt * x                            │
          │                                  │
          └──────────┬───────────────────────┘
                     ▼
           output = weight ⊙ tmp
```

---

## 3. Matmul（矩阵乘法）

**功能**：通用矩阵乘法 `C = α * A × B` (α 默认 1.0)，支持 FP32 和 INT8 量化两种模式。

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/matmul_kernel.cpp

void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, const CudaConfig* config) {
  // input:  [in_dim0, in_dim1]  即 [batch_size, dim]（此处 in_dim1 不是权重的 dim1）
  // 实际通过 Armadillo 表达为: arma::fmat(in_ptr, wei_dim1, batch_size, false, true)
  //
  // weight: [wei_dim0, wei_dim1] 即 [out_dim, in_dim]
  // output: [out_dim, batch_size] (Armadillo 列主序视角)

  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  const float* output_ptr = output.ptr<float>();

  // 动态推断 batch_size: input.size() / dim1_（每个 token embedding 的维度）
  int32_t batch_size = input.size() / wei_dim1;

  // Armadillo 使用列主序，需转置视角
  arma::fmat input_mat(const_cast<float*>(input_ptr), wei_dim1, batch_size, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(const_cast<float*>(output_ptr), wei_dim0, batch_size, false, true);

  output_mat = ((input_mat * weight_mat)) * scale;
}
```

### CUDA 实现（单请求 + batched，同一个 kernel）

```cpp
// 文件: kuiper/source/op/kernels/cuda/matmul_kernel.cu
// 使用 cuBLAS 库，其底层 SGEMM 天然支持 batched 矩阵乘法

// 上层调用 (matmul.cpp):
// kernel::get_matmul_kernel(device_type)(input, weight, output, scale, config);
//
// CUDA 路径通过 cuBLAS 的 cublasSgemm / cublasGemmEx 实现
// 关键参数:
//   - m = dim0_ (输出维度)
//   - n = batch_size (input.size() / dim1_)
//   - k = dim1_ (输入维度)
//   自动适配 [batch_size, dim] × [dim, out_dim] -> [batch_size, out_dim]
```

### 执行流程

```
输入 A: [batch_size, dim]      权重 W: [dim, out_dim] (预先转置好)
         │                              │
         │     ┌────────────────────────┘
         ▼     ▼
    C = A × W × scale
         │
         ▼
输出 C: [batch_size, out_dim]
```

---

## 4. RoPE（旋转位置编码）

**功能**：对 Q 和 K 向量施加旋转位置编码，使注意力计算能感知 token 位置。

### CPU 实现（单请求）

```cpp
// 文件: kuiper/source/op/kernels/cpu/rope_kernel.cpp

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  // 取第一个请求的位置（单请求模式）
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  // 遍历 dim 中的每一对相邻元素
  for (int32_t i = 0; i < dim; i += 2) {
    int32_t head_dim = i % head_size;  // 在注意力头内的偏移

    // 查预计算缓存: 位置 pos、维度 head_dim 对应的旋转角
    float fci = sin_cache[pos * head_size + head_dim];  // sin(θ)
    float fcr = cos_cache[pos * head_size + head_dim];  // cos(θ)

    // rotn=1: 只旋转 Q (当 i >= kv_dim 时，GQA 下 K 维度小于 Q)
    // rotn=2: 同时旋转 Q 和 K
    int32_t rotn = i < kv_dim ? 2 : 1;

    for (int32_t v = 0; v < rotn; v++) {
      float* vec = (v == 0) ? input_q : input_k;  // 选 Q 或 K
      // 2D 旋转: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i]     = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}
```

### CUDA 实现（Batched）

```cpp
// 文件: kuiper/source/op/kernels/cuda/rope_kernel.cu

__global__ void batched_rope_kernel(
    int32_t batch_size, int32_t q_dim, int32_t kv_dim, int32_t head_size,
    const int32_t* pos_array,     // [batch_size] 每个请求的当前位置
    const float* sin_cache, const float* cos_cache,
    float* q_array,               // [batch_size, q_dim]
    float* k_array)               // [batch_size, kv_dim]
{
  int batch_idx = blockIdx.y;          // Y 轴索引：第几个请求
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;  // X 轴索引：特征维度
  if (batch_idx >= batch_size) return;

  int idx = dim_idx * 2;               // 每线程处理一对相邻元素
  if (idx >= q_dim) return;

  // 从 pos_array 获取该请求的位置
  int32_t pos = pos_array[batch_idx];
  int head_dim = idx % head_size;

  // 查 sin/cos 缓存
  float fci = sin_cache[pos * head_size + head_dim];
  float fcr = cos_cache[pos * head_size + head_dim];

  // 旋转 Q
  int q_offset = batch_idx * q_dim + idx;
  rope_calc(fcr, fci, q_array, q_offset);

  // 旋转 K（仅当 idx < kv_dim，处理 GQA）
  if (idx < kv_dim) {
    int k_offset = batch_idx * kv_dim + idx;
    rope_calc(fcr, fci, k_array, k_offset);
  }
}

// Launch: grid(head_pairs, batch_size), block(128)
// grid.y = batch_size 是关键：每个 Y 行对应一个请求
```

### 执行流程

```
单请求:
  pos (标量) + sin/cos_cache[pos] ──▶ 旋转 Q[0:dim] 和 K[0:kv_dim]

Batched:
  pos_array[0] + cache ──▶ 旋转 Q[0, :], K[0, :]     (请求 0)
  pos_array[1] + cache ──▶ 旋转 Q[1, :], K[1, :]     (请求 1)
  ...
  全部并行在同一个 kernel launch 中完成

旋转公式（2D）:
  [x']   [cos(θ)  -sin(θ)] [x]
  [y'] = [sin(θ)   cos(θ)] [y]
```

---

## 5. KV Cache Write（Paged KV 缓存写入）

**功能**：每步推理后，将当前 token 的 K、V 写入 Paged KV Cache 的对应物理块位置。

### CUDA 实现（仅 batched，也是唯一的实现）

```cpp
// 文件: kuiper/source/op/kernels/cuda/kvcache_kernel.cu

__global__ void batched_write_paged_kv_kernel(
    const float* k_src,               // [batch_size, kv_dim] 新计算的 K
    const float* v_src,               // [batch_size, kv_dim] 新计算的 V
    const int32_t* pos_array,         // [batch_size] 当前位置
    const int32_t* block_table_2d,    // [batch_size, max_blocks_per_req] 逻辑→物理块映射
    float* key_cache_pool,            // [num_layers, total_blocks * block_size, kv_dim]
    float* value_cache_pool,
    int32_t batch_size, int32_t kv_dim, int32_t block_size,
    int32_t max_blocks_per_req, int32_t layer_idx, int32_t total_blocks)
{
  int req_idx = blockIdx.y;                       // 第几个请求
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;  // kv_dim 维度索引
  if (req_idx >= batch_size || dim_idx >= kv_dim) return;

  // Step 1: 计算逻辑块和块内偏移
  int32_t pos = pos_array[req_idx];
  int32_t logical_block_idx = pos / block_size;   // 第几个逻辑块
  int32_t token_offset = pos % block_size;         // 块内偏移

  // Step 2: 查 block_table 获取物理块 ID
  const int32_t* block_row = block_table_2d + req_idx * max_blocks_per_req;
  int32_t physical_block_idx = block_row[logical_block_idx];

  // Step 3: 计算物理地址偏移
  // 层偏移 + 物理块偏移 + 块内 token 偏移 + 维度偏移
  int64_t layer_offset = layer_idx * total_blocks * block_size * kv_dim;
  int64_t physical_offset = layer_offset +
      physical_block_idx * block_size * kv_dim +
      token_offset * kv_dim + dim_idx;

  // Step 4: 写入
  int64_t src_offset = req_idx * kv_dim + dim_idx;
  key_cache_pool[physical_offset]   = k_src[src_offset];
  value_cache_pool[physical_offset] = v_src[src_offset];
}

// Launch: grid(ceil(kv_dim/128), batch_size), block(128)
```

### 执行流程

```
                    K_src [batch_size, kv_dim]
                    V_src [batch_size, kv_dim]
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼ req_0                ▼ req_1                ▼ req_{B-1}
 pos_array[0]           pos_array[1]           pos_array[B-1]
       │                       │                       │
  pos / block_size      pos / block_size        pos / block_size
       │                       │                       │
  block_table[0,        block_table[1,          block_table[B-1,
   logic_blk] ──物理块──▶  logic_blk] ──物理块──▶   logic_blk] ──物理块──▶
       │                       │                       │
       ▼                       ▼                       ▼
  key_cache_pool         key_cache_pool          key_cache_pool
  [layer, phys_blk,      [layer, phys_blk,       [layer, phys_blk,
   token_off, :]          token_off, :]           token_off, :]
```

> ⚠️ CPU 路径缺失此算子。CPU MHA kernel 直接读取 KV cache，但写入路径只有 CUDA 分支 (`llama3.cpp:539`)。

---

## 6. Softmax（注意力归一化）

**功能**：对注意力分数做 in-place softmax，用于 MHA 内部。内嵌在 MHA kernel 中。

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/softmax_kernel.cpp

void softmax_inplace_cpu(const tensor::Tensor& input, void* stream) {
  int32_t size = static_cast<int32_t>(input.size());
  const float* input_ptr = input.ptr<float>();

  // Step 1: 找最大值（数值稳定性）
  float max_value = *std::max_element(input_ptr, input_ptr + size);

  // Step 2: exp(x - max) 并累加 sum
  arma::fvec input_mat(const_cast<float*>(input_ptr), size, false, true);
  input_mat = arma::exp(input_mat - max_value);
  float sum_value = arma::sum(input_mat);

  // Step 3: 除以 sum
  input_mat = input_mat / sum_value;
}
```

### CUDA 实现（GPU block 内 softmax，内嵌于 MHA kernel）

```cpp
// 文件: kuiper/source/op/kernels/cuda/mha_kernel.cu

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // Phase 1: 找最大值 — 跨线程 Reduce(Max)
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) max_val = x[i];
  }
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  __syncthreads();

  // Phase 2: exp(x - max) + 求和 — 跨线程 Reduce(Sum)
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  __syncthreads();

  // Phase 3: 归一化
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}
```

### 执行流程

```
输入 scores: [0 .. pos]（当前上下文长度个注意力分数）
                           │
  ┌────────────────────────┼────────────────────────┐
  ▼                        ▼                        ▼
max = max(scores)    exp(s_i - max)           sum = Σ exp(s_i - max)
                           │                        │
                           └────────┬───────────────┘
                                    ▼
                           probs[i] = exp(s_i - max) / sum
```

---

## 7. Multi-Head Attention（多头注意力，Paged）

**功能**：执行 Paged Multi-Head Attention。每个 head 内：`Softmax(Q × K^T / √d) × V`，其中 K、V 从非连续的 Paged 缓存中按 block_table 查找。

### CPU 实现（单请求）

```cpp
// 文件: kuiper/source/op/kernels/cpu/mha_kernel.cpp

void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                const tensor::Tensor& mha_out, ..., const tensor::Tensor& block_table_tensor,
                int32_t block_size, base::DeviceType device_type, CudaConfig* config) {

  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float scale = 1.f / sqrt(head_size);
  int32_t* block_table = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());

  // 逐 head 串行处理
  for (int32_t h = 0; h < head_num; ++h) {
    float* query_head = query_tensor.ptr<float>() + h * head_size;   // Q_h: [head_size]
    float* score_head = score_tensor.ptr<float>() + h * seq_len;     // S_h: [seq_len]
    int32_t head_offset = (h / kv_mul) * head_size;  // GQA: 多个 Q head 共享一个 KV head

    // === 阶段 1: Q × K^T（逐 token 迭代） ===
    for (int32_t t = 0; t <= pos; t++) {
      // 查表定位物理地址
      int logical_block_idx = t / block_size;
      int token_offset = t % block_size;
      int physical_block_idx = block_table[logical_block_idx];

      int64_t physical_offset = layer_offset +
          physical_block_idx * block_size * kv_dim +
          token_offset * kv_dim + head_offset;

      // K_h[t] 与 Q_h 点乘
      float* key_head = key_cache_pool + physical_offset;
      score_head[t] = dot_product(query_head, key_head, head_size) * scale;
    }

    // === 阶段 2: Softmax ===
    softmax_inplace_cpu(score_head, pos + 1);

    // === 阶段 3: Score × V ===
    float* output_head = mha_out + h * head_size;
    memset(output_head, 0, head_size * sizeof(float));
    for (int32_t t = 0; t <= pos; t++) {
      // 再次查表获取 V_h[t]
      int physical_block_idx = block_table[t / block_size];
      int64_t physical_offset = layer_offset +
          physical_block_idx * block_size * kv_dim +
          (t % block_size) * kv_dim + head_offset;
      float* value_head = value_cache_pool + physical_offset;
      // output_head[i] += score[t] * value_head[i]
      for (int32_t i = 0; i < head_size; i++) {
        output_head[i] += score_head[t] * value_head[i];
      }
    }
  }
}
```

### CUDA 实现（Batched）

```cpp
// 文件: kuiper/source/op/kernels/cuda/mha_kernel.cu

__global__ void batched_paged_multi_head_attention_kernel(
    const int32_t* pos_array,       // [batch_size]
    int32_t seq_len,
    float* query,                   // [batch_size, head_num * head_size]
    float* score_ptr,               // [batch_size, head_num, seq_len]
    float* output,                  // [batch_size, head_num * head_size]
    float* key_cache_pool,          // [num_layers, total_blocks * block_size, kv_dim]
    float* value_cache_pool,
    const int32_t* block_table_2d,  // [batch_size, max_blocks_per_req]
    int32_t block_size, int32_t max_blocks_per_req,
    int32_t kv_dim, int32_t kv_mul, int32_t head_num, int32_t head_size,
    int32_t layer_idx, int32_t total_blocks_pool)
{
  int head = blockIdx.x;         // 哪个注意力头
  int batch_idx = blockIdx.y;   // 哪个请求

  // 取该请求的独有参数
  int32_t pos = pos_array[batch_idx];
  const int32_t* block_table = block_table_2d + batch_idx * max_blocks_per_req;
  int64_t layer_offset = layer_idx * total_blocks_pool * block_size * kv_dim;

  // 指针按 batch 和 head 偏移
  float* query_head = query + batch_idx * (head_num * head_size) + head * head_size;
  float* score_head = score_ptr + batch_idx * (head_num * seq_len) + head * seq_len;
  float* output_head = output + batch_idx * (head_num * head_size) + head * head_size;
  int head_offset = (head / kv_mul) * head_size;

  // 加载 Q 到共享内存（减少全局内存访问）
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  // === 阶段 1: Q × K^T ===
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    int physical_block_idx = block_table[t / block_size];
    int64_t phys_off = layer_offset +
        physical_block_idx * block_size * kv_dim +
        (t % block_size) * kv_dim + head_offset;

    float* key_head = key_cache_pool + phys_off;
    // float4 向量化点积，每条指令计算 4 个乘加
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *(float4*)(key_head + i);
      float4 query_val = *(float4*)(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y +
               key_val.z * query_val.z + key_val.w * query_val.w;
    }
    score_head[t] = score * scale;
  }
  __syncthreads();

  // === 阶段 2: Softmax ===
  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  // === 阶段 3: Score × V ===
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      int physical_block_idx = block_table[t / block_size];
      int64_t phys_off = layer_offset +
          physical_block_idx * block_size * kv_dim +
          (t % block_size) * kv_dim + head_offset;
      value += score_head[t] * value_cache_pool[phys_off + i];
    }
    output_head[i] = value;
  }
}

// Launch: grid(head_num, batch_size), block(256), shmem(head_size * sizeof(float))
```

### 执行流程

```
对于每个 (head, batch_item):
  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  Q_h [head_size]     K_cache (Paged, 非连续)              │
  │       │                      │                            │
  │       │    ┌─────────────────┘                            │
  │       ▼    ▼                                              │
  │   Q × K^T:  对 t = 0..pos:                                │
  │     block_id = block_table[t / block_size]   // 查表      │
  │     phys_addr = layer_off + block_id * block_sz * kv_dim  │
  │                + (t % block_sz) * kv_dim + head_off       │
  │     score[t] = dot(Q_h, K_cache[phys_addr]) * 1/√d        │
  │       │                                                   │
  │       ▼                                                   │
  │   Softmax(score[0..pos])                                  │
  │       │                                                   │
  │       ▼                                                   │
  │   Score × V:  对 i = 0..head_size-1:                       │
  │     output[i] = Σ_{t=0..pos} score[t] * V[phys_addr + i]  │
  │                                                           │
  └───────────────────────────────────────────────────────────┘

Grid 结构:
  GPU block (head=2, batch_idx=1): 处理请求1的第2个注意力头
  GPU block (head=0, batch_idx=0): 处理请求0的第0个注意力头
  ...
  全部 head_num × batch_size 个 block 并行执行
```

---

## 8. SwiGLU（门控激活）

**功能**：SwiGLU 激活函数，用于 FFN 层。`output = sigmoid(w1_out) ⊙ w1_out ⊙ w3_out`

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/swiglu_kernel.cpp

void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
  // input1: w1 输出 [hidden_dim] 或 [batch_size, hidden_dim]
  // input2: w3 输出 [hidden_dim] 或 [batch_size, hidden_dim]

  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

  // Step 1: input1 = sigmoid(input1) = 1 / (1 + exp(-input1))
  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));

  // Step 2: output = sigmoid(input1) ⊙ input2 (逐元素乘法)
  output_vec = input1_vec % input2_vec;
}
```

### CUDA 实现

```cpp
// 文件: kuiper/source/op/kernels/cuda/swiglu_kernel.cu

__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;

  // 共享内存缓冲（双缓冲 in1 和 in2）
  extern __shared__ float shared_mem[];
  float* smem1 = shared_mem;
  float* smem2 = shared_mem + blockDim.x;

  smem1[tid] = in1[idx];
  smem2[tid] = in2[idx];
  __syncthreads();

  // sigmoid(in1[tid]) * in1[tid] * in2[tid]
  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;    // sigmoid(in1) * in1 = SiLU(in1)
  out[idx] = smem1[tid] * smem2[tid]; // × gate
}

// Launch: grid(ceil(size/128), 1), block(128), shmem(128 * sizeof(float) * 2)
```

> 单请求和 batched 使用**同一个 kernel**。区别仅在于 `size = hidden_dim` vs `size = batch_size * hidden_dim`。CUDA kernel 做的是逐元素操作，天然适配任意大小。

### 执行流程

```
w1_out [hidden_dim]          w3_out [hidden_dim]
       │                             │
       ▼                             │
  SiLU(w1_out)                       │
  = w1_out × σ(w1_out)               │
       │                             │
       └──────────┬──────────────────┘
                  ▼
        output = SiLU(w1_out) ⊙ w3_out  (逐元素乘法)
```

---

## 9. VecAdd（残差连接）

**功能**：逐元素加法 `output = input1 + input2`，实现残差连接。

### CPU 实现

```cpp
// 文件: kuiper/source/op/kernels/cpu/add_kernel.cpp

void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  // Armadillo 向量化加法
  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
  output_vec = input_vec1 + input_vec2;
}
```

### CUDA 实现

```cpp
// 文件: kuiper/source/op/kernels/cuda/add_kernel.cuh
// add_kernel_cu —— 逐元素向量加法，单 kernel launch

// 同样天然支持任意大小，单请求为 [dim]，batched 为 [batch_size, dim]
```

### 执行流程

```
input1 [N]     input2 [N]
     │              │
     └──────┬───────┘
            ▼
     output[i] = input1[i] + input2[i]   (∀ i ∈ [0, N))

在 Llama 推理中的使用场景:
  - attention 后:  x = x + attn_output      (残差连接)
  - FFN 后:       x = x + ffn_output       (残差连接)
```

---

## 10. Argmax Sampler（贪心采样）

**功能**：从 logits 向量中取最大值的索引，作为下一个 token ID。

### 实现

```cpp
// 文件: kuiper/source/sampler/argmax_sampler.cpp

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // CPU: std::max_element 直接找最大值下标
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    // GPU: 调用 CUDA argmax kernel
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
  }
}
```

### 执行流程

```
logits [vocab_size]  (例如 151936 维)
        │
        ▼
  argmax: 找到最大值的索引
        │
        ▼
  next_token_id
```

> Batched 模式下，在 `generate_batch()` 的调用层做循环：每个请求单独调用 sampler，从 `forward_output[b * vocab_size : (b+1) * vocab_size]` 切片中采样。

---

## 附录：单层 LLaMA 完整推理流水线

```
输入: tokens [batch_size], positions [batch_size]

1. Embedding
   tokens ──查表──▶ hidden [batch_size, dim]

   for layer_idx = 0 .. num_layers-1:

2.   RMSNorm (Pre-Attention)
     hidden ──RMS──▶ normed [batch_size, dim]

3.   QKV 投影
     normed × Wq ──▶ Q [batch_size, dim]
     normed × Wk ──▶ K [batch_size, kv_dim]
     normed × Wv ──▶ V [batch_size, kv_dim]

4.   RoPE (位置编码)
     Q, K ──旋转──▶ Q_rope, K_rope

5.   KV Cache Write (Paged)
     K_rope, V ──写入──▶ K_cache[blocks], V_cache[blocks]

6.   Multi-Head Attention (Paged)
     Q_rope × K_cache^T ──▶ Scores ──Softmax──▶ AttnWeights
     AttnWeights × V_cache ──▶ AttnOutput
     AttnOutput × Wo ──▶ attn_result [batch_size, dim]

7.   Residual Add (Post-Attention)
     hidden = hidden + attn_result

8.   RMSNorm (Pre-FFN)
     hidden ──RMS──▶ ffn_normed [batch_size, dim]

9.   FFN (SwiGLU)
     ffn_normed × W1 ──▶ gate [batch_size, hidden_dim]
     ffn_normed × W3 ──▶ up   [batch_size, hidden_dim]
     SiLU(gate) ⊙ up ──▶ activated
     activated × W2 ──▶ ffn_result [batch_size, dim]

10.  Residual Add (Post-FFN)
     hidden = hidden + ffn_result

   end for

11. RMSNorm (Final)
    hidden ──RMS──▶ final_normed [batch_size, dim]

12. LM Head
    final_normed × W_lm_head ──▶ logits [batch_size, vocab_size]

13. Sampling
    logits ──Argmax──▶ next_tokens [batch_size]
```

---

## 附录：算子实现矩阵

| 算子 | CPU | CUDA 单请求 | CUDA Batched | 文件 |
|------|:---:|:----------:|:------------:|------|
| Embedding | ✅ | ✅ | ✅ (天然) | `emb_kernel.{cpp,cu}` |
| RMSNorm | ✅ | ✅ | ⚠️ (逐item循环) | `rmsnorm_kernel.{cpp,cu}` |
| Matmul | ✅ | ✅ | ✅ (天然) | `matmul_kernel.{cpp,cu}` |
| RoPE | ✅ | ✅ | ✅ | `rope_kernel.{cpp,cu}` |
| KV Cache Write | ❌ | ❌ | ✅ (唯一实现) | `kvcache_kernel.cu` |
| Softmax | ✅ | ✅ | ✅ (内嵌MHA) | `softmax_kernel.cpp` / `mha_kernel.cu` |
| Paged MHA | ✅ (单请求) | ✅ | ✅ | `mha_kernel.{cpp,cu}` |
| SwiGLU | ✅ | ✅ | ✅ (天然) | `swiglu_kernel.{cpp,cu}` |
| VecAdd | ✅ | ✅ | ✅ (天然) | `add_kernel.{cpp,cuh}` |
| Argmax Sampler | ✅ | ✅ | ❌ (逐item调用) | `argmax_sampler.cpp` / `argmax_kernel.cuh` |

- ✅ 完整实现
- ⚠️ 有实现但需上层循环调用
- ❌ 未实现
- "天然" = kernel 按逐元素/矩阵乘法模式设计，`size` 参数扩展即可适配 batch
