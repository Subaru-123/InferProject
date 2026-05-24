# Continuous Batching 完整改造方案

> 本文档详细列出在现有 Static Batching 基础上实现真正 Continuous Batching 所需的**全部修改**，按模块和优先级组织。

---

## 总览

```
现有管线:  请求集合固定 → 同步步进 → 结束
目标管线:  请求队列 → 调度器 → 每步动态组装 batch → 完成回收 + 新请求注入
```

| 模块 | 改造量 | 说明 |
|------|--------|------|
| 1. 调度器 (Scheduler) | **新增** | 请求队列 + 准入控制 + 优先级 |
| 2. Prefill/Decode 分离 | **改造 forward** | prefill 一次处理多 token，decode 逐 token |
| 3. 主循环重构 | **重写 generate_batch** | 动态 batch + 多阶段混合执行 |
| 4. 请求状态管理 | **新增 ReqState** | 每请求独立位置、token 列表、KV block 记录 |
| 5. Block 生命周期管理 | **改造 BlockManager** | 请求完成 → 释放 block → 回到池中复用 |
| 6. Batch 动态组装 (Gather) | **新增** | 从活跃请求拼装当前 step 的 batch 输入 |
| 7. 结果分发 (Scatter) | **新增** | 采样结果写回各请求，标记完成/续推 |
| 8. Buffer 管理适配 | **改造 init_mem** | buffer 预分配不变，运行时按实际 batch 切片 |
| 9. RMSNorm 切片策略 | **改造 attention_rms 等** | 目前已经做了逐 item 切片，保持不变 |
| 10. CPU 路径补齐 | **改造** | 补充 KV Cache 写入 + CPU MHA batched 版 |
| 11. 性能优化 | **可选** | CUDA Graph、算子融合、异步 overlap |

---

## 1. 调度器（新增）

### 现状
无。请求在 `main.cpp` 中一次性创建，batch 组成固定不变。

### 需新增的类

```cpp
// 新建文件: kuiper/include/base/scheduler.h

enum class ReqStatus { kWaiting, kPrefilling, kDecoding, kFinished };

struct ScheduleRequest {
    int32_t req_id;
    std::vector<int32_t> prompt_tokens;  // 完整 prompt
    int32_t prompt_len;
    int32_t pos = 0;                     // 当前处理到的位置
    ReqStatus status = ReqStatus::kWaiting;
    std::vector<int32_t> block_ids;      // 已分配的物理块 ID 列表
    std::vector<int32_t> generated_ids;  // 生成的 token
    int32_t max_gen_len = 256;
    double arrival_time;                 // 到达时间戳
};

class Scheduler {
public:
    // 提交新请求到等待队列
    int32_t submit(const std::vector<int32_t>& prompt_tokens, int32_t max_gen_len);

    // 核心：每步调用，返回本步该执行的请求列表
    // 同时处理: 新请求准入(prefill) / 进行中请求续推(decode) / 完成请求回收
    struct StepBatch {
        std::vector<int32_t> active_indices;   // 本轮活跃的 req_id 列表
        std::vector<bool> is_prefill;           // 每项是 prefill 还是 decode
        int32_t total_batch_size;
    };
    StepBatch schedule_step();

    // 标记请求完成，释放其占用的 blocks
    void finish_request(int32_t req_id, BlockManager& block_manager);

private:
    std::deque<ScheduleRequest> waiting_queue_;   // 等待队列
    std::unordered_map<int32_t, ScheduleRequest> active_requests_;  // 活跃请求
    int32_t max_active_requests_ = 64;
    int32_t max_prefill_tokens_per_step_ = 2048;  // 单步最多处理多少 prefill token
    int32_t next_req_id_ = 0;
};
```

### 调度策略

```
每步 schedule_step() 的决策逻辑:

1. 统计当前 decode 请求数
2. 计算剩余 slot = max_active_requests_ - decode 数
3. 从 waiting_queue_ 中取出 ≤ slot 个请求进入 prefill
4. 限制: 本步 prefill token 总数 ≤ max_prefill_tokens_per_step_
5. 返回 StepBatch{ active_indices[], is_prefill[] }

关键约束:
- decode 请求永远优先（不能让正在生成的请求饿死）
- prefill 的总 token 数有上限（避免单步延迟过高）
- 没有 free slot 或 block 不够时，新请求留在 waiting_queue_
```

---

## 2. Prefill / Decode 阶段分离

### 现状
`generate_batch` 中对每个请求逐 token 调用 `embedding()` + `forward()`。即使用同样的 1-token 路径处理 prompt token 和生成 token。

### 问题

对于 prefill，已知所有 prompt token，逐 token 走 `forward` 严重浪费矩阵乘法的并行度：

```
逐 token:   [1, dim] × [dim, out_dim] → batch=1，GPU 利用率极低
批量 prefill: [prompt_len, dim] × [dim, out_dim] → 大矩阵乘法，GPU 满载
```

但 prefill 和 decode 不能完全统一用一个 `forward`——因为 prefill 不需要采样中间结果（teacher forcing），KV Cache 的写入模式也不同。

### 需改造的地方

**方案：拆分两个 forward 路径**

```cpp
// llm中新增两个方法

// Prefill: 一次性处理所有 prompt token
// input: [prompt_len, dim], pos_tensor: [prompt_len] (递增: 0,1,2,...)
// 返回: 仅最后一个位置的 logits
base::Status forward_prefill(
    const tensor::Tensor& input,       // [prefill_tokens, dim]
    const tensor::Tensor& pos_tensor,  // [prefill_tokens]
    int& next);

// Decode: 逐 token 生成 (与当前 forward 完全一致)
// input: [batch_size, dim], pos_tensor: [batch_size]
base::Status forward_decode(
    const tensor::Tensor& input,
    const tensor::Tensor& pos_tensor,
    int& next);
```

**Prefill 与 Decode 的关键差异：**

| 维度 | Prefill | Decode |
|------|---------|--------|
| 每次处理的 token 数 | 多个（prompt_len） | 1 个（batch_size 个请求各 1 token） |
| KV Cache 写入 | 连续写入多个位置 | 每个请求写 1 个位置 |
| 是否需要中间采样 | 不需要（teacher forcing） | 需要 |
| 输出 | 只取最后一个位置的 logits | 每个请求取 1 个 logits 行 |
| Matmul 效率 | 高（大矩阵） | 低（batch_size × 小矩阵） |

**实际实现时可以选择简化方案：**

由于 prefill 和 decode 在同一个 step 内混合发生（新请求做 prefill，老请求做 decode），最简单的处理是：

```
每步:
  1. 将所有请求的当前 token 拼成一个 batch
  2. pos_array 标记每个请求的当前位置（prefill 和 decode 各有不同的 pos）
  3. 走同一个 forward()
  4. 采样：只有 pos >= prompt_len-1 的请求才采样
```

这种方式**不需要修改 forward()**，只需要在主循环中正确拼装 batch 输入即可。代价是 prefill 阶段没有利用大矩阵乘法的优势，但实现简单、验证快。

更优的做法是 prefill 阶段利用 embedding + matmul 一次处理多个 prompt token（见 "优化方向" 节）。

---

## 3. 主循环重构

### 现状（main.cpp generate_batch）

```cpp
// 伪代码
batch_size = sentences.size();      // 固定
pos = 0;                            // 所有请求共享

while (pos < 128) {
    for (b in 0..batch_size) {
        token = (pos < prompt_len[b]) ? prompt_token[b][pos] : next_token[b];
        pos_tensor[b] = pos;        // 所有人同一个 pos！
    }
    embedding(tokens);              // batch 固定
    forward(input, pos_tensor);
    采样(仅 pos >= prompt_len-1 的请求);
    pos++;                          // 全局步进
}
```

### 重构后

```cpp
int32_t generate_continuous_batching(
    model::LLama2Model& model,
    Scheduler& scheduler,
    int32_t max_total_steps)
{
    std::vector<int32_t> pos_array(model.max_batch_size_, 0);
    std::vector<int32_t> block_ids_per_req(model.max_batch_size_ * model.max_blocks_per_req_, -1);
    // ... 初始化

    int32_t step = 0;
    while (step < max_total_steps) {
        // ─── Phase 1: 调度 ───
        auto step_batch = scheduler.schedule_step();  // 获取本轮活跃请求
        if (step_batch.active_indices.empty()
            && scheduler.waiting_queue_empty()) break;

        int32_t cur_batch = step_batch.active_indices.size();
        // 如果 batch 太小，补充预取（可选优化）

        // ─── Phase 2: Gather ───
        std::vector<int32_t> cur_tokens(cur_batch);
        for (int b = 0; b < cur_batch; ++b) {
            auto& req = scheduler.get_request(step_batch.active_indices[b]);
            if (req.status == kPrefilling) {
                // prefill: 取当前 pos 处的 prompt token
                cur_tokens[b] = req.prompt_tokens[req.pos];
            } else {
                // decode: 取上一次生成的 token
                cur_tokens[b] = req.generated_ids.back();
            }
            pos_array[b] = req.pos;
        }

        // ─── Phase 3: Forward ───
        auto emb = model.embedding(cur_tokens);
        model.forward(emb.input_embeddings, pos_tensor, dummy_next);
        cudaDeviceSynchronize();

        // ─── Phase 4: Scatter ───
        const float* logits = model.get_buffer(kForwardOutput).ptr<float>();
        int32_t vocab_size = std::abs(model.config_->vocab_size_);

        for (int b = 0; b < cur_batch; ++b) {
            auto& req = scheduler.get_request(step_batch.active_indices[b]);

            if (req.status == kPrefilling && req.pos < req.prompt_len - 1) {
                // prefill 中间步骤: 不需要采样, 直接用下一个 prompt token
                req.pos++;
                continue;
            }

            // 采样
            int32_t next = argmax_kernel_cu(logits + b * vocab_size, vocab_size, nullptr);
            req.generated_ids.push_back(next);
            req.pos++;

            // 检查终止
            if (model.is_sentence_ending(next)
                || req.generated_ids.size() >= req.max_gen_len) {
                scheduler.finish_request(req.req_id, model.block_manager_);
            }
        }

        step++;
    }
}
```

### 关键变化点

| 变化 | 说明 |
|------|------|
| `pos` | 从全局 scalar → 每请求独立的 `pos_array` |
| `batch_size` | 从固定 → 每步动态变化 |
| prefill 中间步骤 | 不采样，直接 teacher-forcing 下一个 prompt token |
| 请求结束 | 调用 `scheduler.finish_request()` 回收 blocks |
| `is_finished` | 不再需要（完成即移出活跃列表） |

---

## 4. 请求状态管理（新增）

### 现状
请求状态散落在 `main.cpp` 的多个 `std::vector` 中（`prompt_lens`, `tokens_batch`, `next_tokens`, `is_finished`, `words_batch`），结构不清晰。

### 需新增

```cpp
// 统一请求状态，替代散落的多个 vector
struct ReqState {
    int32_t req_id;
    std::vector<int32_t> prompt_tokens;   // 完整 prompt
    std::vector<int32_t> generated_tokens; // 已生成的 token
    int32_t pos = 0;                       // 当前处理到的位置
    int32_t prompt_len;                    // prompt 长度
    bool is_prefill = true;                // 是否还在 prefill 阶段
    bool is_finished = false;
    std::vector<int32_t> block_ids;        // 占用的物理块 ID（用于回收）
};
```

这条改动很小但很重要——它把状态从"过程式变量"变成了"结构化对象"，后续所有模块都基于它工作。

---

## 5. Block 生命周期管理

### 现状

```cpp
// init_mem() 一次性分配 total = max_batch_size × max_blocks_per_req 个块
// ensure_batch_blocks() 按需分配，但从不回收
// block_manager 只有 allocate, 没有调用 free (代码有 free 函数, 但没有调用点)
```

### 需改造

**5.1 记录每个请求占用的物理块**

```cpp
// Scheduler::finish_request() 中:
void Scheduler::finish_request(int32_t req_id, BlockManager& bm) {
    auto& req = active_requests_[req_id];
    // 回收该请求占用的所有物理块
    bm.free_blocks(req.block_ids);
    req.block_ids.clear();
    req.status = ReqStatus::kFinished;
    active_requests_.erase(req_id);
}
```

**5.2 新请求的 Block 分配**

```cpp
// Scheduler 准入新请求时:
int32_t Scheduler::admit_request(ScheduleRequest& req, BlockManager& bm) {
    int32_t blocks_needed = (req.prompt_len + max_gen_len + block_size - 1) / block_size;
    if (bm.free_block_count() < blocks_needed) {
        return -1;  // 块不够，拒绝准入（或触发抢占）
    }
    for (int i = 0; i < blocks_needed; ++i) {
        req.block_ids.push_back(bm.allocate_block());
    }
    return 0;
}
```

**5.3 抢占（可选，进阶）**

当块池耗尽而有高优先级请求到达时，选择低优先级请求驱逐，释放其块给新请求。大规模场景才需要。

---

## 6. Batch 动态组装（Gather）

### 现状
`ensure_batch_blocks()` 假设 batch 组成不变——它只检查未分配的 slot 并分配新块，但从不释放旧块。

### 需改造

Batch 组装从"固定 index 对固定请求"变为"slot 复用"：

```
Step N:   slot_0: req_A,  slot_1: req_B,  slot_2: req_C
Step N+1: slot_0: req_D,  slot_1: req_B,  slot_2: req_E  ← A 完成释放, C 完成释放
```

**每次 batch 组成变化时需要做：**

1. 重排 `block_table` 的行（将新请求的 block 映射写入对应 slot）
2. `cudaMemcpyAsync` 将新的 `block_table` 同步到 GPU
3. 构建 `cur_tokens` 和 `pos_array` 时从新的活跃请求读取

关键约束：**batch size ≤ max_batch_size_**（由 `init_mem` 预分配决定），这是硬上限。

---

## 7. 结果分发（Scatter）

### 现状
采样后立即更新 `next_tokens[b]` 和 `words_batch[b]`，标记 `is_finished[b]`。

### 需改造

```
采样后:
  对每个活跃 slot:
    if (is_prefill && pos < prompt_len - 1):
        → 不采样，pos++, 用下一个 prompt token 作为 next
    else:
        → 采样得到 next_token
        → 加入 req.generated_ids
        → pos++
        → 检测终止:
            if (终止符 或 达到 max_gen_len):
                → scheduler.finish_request(req_id, block_manager)
                  // 释放该请求的所有 blocks
```

---

## 8. Buffer 管理适配

### 现状
`init_mem()` 按 `max_batch_size_` 预分配所有 buffer。这是正确的——CB 不需要更多显存。

### 需注意

- `pos_tensor.reshape({cur_batch_size})` 每步重新 reshape
- `forward_output` 仍然分配 `max_batch_size_ × vocab_size`，但只有前 `cur_batch_size` 行有效
- `score_storage` 分配 `max_batch_size_ × head_num × seq_len`，同理
- `block_table` 恒为 `[max_batch_size_, max_blocks_per_req_]`，行数不变

**核心原则：所有 buffer 大小 = 峰值（max_batch_size_），每步只用前 cur_batch_size 行。不需要动态分配/释放。**

---

## 9. RMSNorm 切片策略

### 现状

```cpp
// attention_rms 中已经有逐 batch item 切片的代码:
for (int i = 0; i < batch_size; ++i) {
    tensor::Tensor in_view(input.dtype(), config_->dim_, false, ...);
    rmsnorm_layer->forward(in_view, out_view);
}
```

### Continuous Batching 下的处理

**不需要额外修改。** 因为当前已经是逐 item 切片调用，与 batch 组成是否固定无关。可变 batch 下只需确保 `batch_size = cur_batch_size`（实际活跃请求数），代码逻辑完全相同。

同理 `feed_forward` 中的 FFN RMSNorm 切片和 `cls_logits` 中的 final RMSNorm 切片都是安全的。

---

## 10. CPU 路径补齐

### 现状
三个缺口：
1. `attention_qkv` 中 KV cache 写入仅 CUDA 分支（`llama3.cpp:539`）
2. CPU MHA kernel 是单请求（`mha_kernel.cpp` 入参为标量 `pos`）
3. CPU RoPE kernel 读取 `*input_pos.ptr<int32_t>(0)`（仅第一个请求的位置）

### 需改造

**10.1 CPU KV Cache 写入**

```cpp
// 在 attention_qkv 中添加 else 分支:
if (device_type_ == base::DeviceType::kDeviceCUDA) {
    kernel::batched_write_paged_kv_cu(...);
} else {
    // CPU 路径: 逐请求逐维度写入 paged KV cache
    for (int b = 0; b < batch_size; ++b) {
        int32_t pos = pos_tensor.index<int32_t>(b);
        int32_t log_blk = pos / block_size_;
        int32_t tok_off = pos % block_size_;
        int32_t phys_blk = single_req_block_table_host_[b * max_blocks_per_req_ + log_blk];
        int64_t layer_off = layer_idx * total_blocks * block_size_ * kv_dim;
        int64_t phys_off = layer_off + phys_blk * block_size_ * kv_dim + tok_off * kv_dim;
        for (int d = 0; d < kv_dim; ++d) {
            key_cache[phys_off + d] = key_src[b * kv_dim + d];
            val_cache[phys_off + d] = val_src[b * kv_dim + d];
        }
    }
}
```

**10.2 CPU MHA Kernel 升级为 Batched**

参照 GPU `batched_paged_multi_head_attention_kernel` 的并行化思路：外层加一层 `batch_idx` 循环，每请求独立查 block_table 和 pos。

**10.3 CPU RoPE Kernel 读取 pos_array**

当前只读 `pos_array[0]`，改为通过参数传入 `batch_idx` 或循环处理所有请求。

---

## 11. 性能优化（可选，后续阶段）

| 优化项 | 说明 | 优先级 |
|--------|------|--------|
| Prefill 批量处理 | prefill 时一次性喂入所有 prompt token，利用大矩阵乘法 | 高 |
| CUDA Graph | 对 decode 阶段的固定 kernel 序列做 graph capture | 中 |
| KV Cache 异步 overlap | 在 GPU 执行 MHA 时，CPU 预取下一步的 block_table 并异步上传 | 中 |
| Flash Attention | 将 Q×K^T 和 softmax×V 融合为单 kernel，减少 HBM 读写 | 低 |
| 算子融合 | RMSNorm + QKV 投影融合为一个 kernel | 低 |

---

## 改造优先级与阶段划分

### Phase 1: 最小可行 CB（~3-5 天）

**目标：能跑通动态 batch，请求可以中途加入和退出。**

1. 实现 `ReqState` 结构体
2. 实现 `Scheduler` 基础版（FCFS + 简单的 slot 管理）
3. 重写 `generate_batch` 主循环（Gather / Forward / Scatter 三阶段）
4. 在 `finish_request` 中调用 `block_manager.free_blocks()` 回收
5. 新请求准入时分配 blocks

**Phase 1 之后：** 功能上已经是真正的 Continuous Batching。

### Phase 2: Prefill 优化（~2-3 天）

1. 拆分 `forward_prefill`（批量处理 prompt token）
2. Scheduler 支持 prefill/decode 混合调度（prefill batch + decode batch 拼在一起）

### Phase 3: 健壮性（~2 天）

1. Block 池耗尽时的处理（拒绝准入 + 等待通知）
2. 请求超时取消
3. CPU 路径补齐

### Phase 4: 性能调优（持续）

1. CUDA Graph
2. Async overlap
3. 性能 benchmark

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| **新增** | `kuiper/include/base/scheduler.h` | 调度器 + ReqState + ReqStatus |
| **新增** | `kuiper/source/base/scheduler.cpp` | 调度器实现 |
| **重写** | `demo/main.cpp` | 新的 `generate_continuous_batching` 主循环 |
| **修改** | `kuiper/source/model/llama3.cpp` | `attention_qkv`: 补充 CPU KV cache 写入分支 |
| **修改** | `kuiper/source/op/kernels/cpu/mha_kernel.cpp` | 升级为 batched：新增外层 batch 循环 + pos_array 参数 |
| **修改** | `kuiper/source/op/kernels/cpu/rope_kernel.cpp` | 支持 pos_array（当前只取 pos_array[0]） |
| **修改** | `kuiper/source/op/mha.cpp` | CPU MHA 调用改为传入 pos_tensor（不再是标量 pos_） |
| **不修改** | `kuiper/source/op/matmul.cpp` | 已支持动态 batch_size 推断 |
| **不修改** | `kuiper/source/op/rope.cpp` | 已支持 batched（dispatch 到 batched_rope_kernel_cu） |
| **不修改** | `kuiper/source/op/kernels/cuda/*.cu` | 所有 CUDA kernel 已支持 batched |
