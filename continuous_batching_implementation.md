# Continuous Batching 实现文档

> 本文档汇总了在 Static Batching 基础上实现真正 Continuous Batching 所做的全部代码修改，包含详细注释。

---

## 修改总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `kuiper/include/base/scheduler.h` | **新增** | Scheduler 类 + ScheduleRequest 状态机 |
| `kuiper/source/base/scheduler.cpp` | **新增** | 调度器方法实现 |
| `demo/main.cpp` | **重写** | CB 主循环 + 动态提交 + block 生命周期 |

---

## 1. Scheduler 调度器

### 1.1 数据结构定义 (`kuiper/include/base/scheduler.h`)

```cpp
#include <base/block_manager.h>
#include <deque>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────
// 请求生命周期状态机:
//   kWaiting → kPrefilling → kDecoding → kFinished
// ─────────────────────────────────────────────────
enum class ReqStatus { kWaiting, kPrefilling, kDecoding, kFinished };

// ─────────────────────────────────────────────────
// 每个请求的完整状态
//   - prompt_tokens: 原始 prompt 的 token 序列
//   - pos: 当前处理到的 token 位置（每个请求独立）
//   - status: prefill/decode 阶段标记
//   - block_ids: 占用的物理块（用于完成时回收）
//   - generated_ids: 生成的 token 序列
// ─────────────────────────────────────────────────
struct ScheduleRequest {
    int32_t req_id;
    std::vector<int32_t> prompt_tokens;  // 完整 prompt token 列表
    int32_t prompt_len;
    int32_t pos = 0;                     // ★ 每个请求独立的位置
    ReqStatus status = ReqStatus::kWaiting;
    std::vector<int32_t> block_ids;      // 已分配物理块 ID 列表
    std::vector<int32_t> generated_ids;  // 已生成 token 列表
    int32_t max_gen_len = 256;
    double arrival_time;
};

// ─────────────────────────────────────────────────
// Scheduler: 管理请求队列和每步 batch 组成
// ─────────────────────────────────────────────────
class Scheduler {
public:
    // 提交新请求到等待队列，返回分配的 req_id
    int32_t submit(const std::vector<int32_t>& prompt_tokens,
                   int32_t max_gen_len);

    // 每步调用：返回本轮活跃请求的 StepBatch
    //   同时完成: 收集活跃请求 + 从等待队列准入新请求
    struct StepBatch {
        std::vector<int32_t> active_indices;   // 本轮活跃 req_id 列表
        std::vector<bool> is_prefill;          // 每个是 prefill 还是 decode
        int32_t total_batch_size;
    };
    StepBatch schedule_step();

    // 标记请求完成，释放其占用的物理块
    void finish_request(int32_t req_id,
                        base::BlockManager& block_manager);

    // 访问器
    ScheduleRequest& get_request(int32_t req_id);
    bool waiting_queue_empty() const;
    int32_t active_count() const;

    // 为新准入 prefill 请求预分配物理块
    //   返回 false 表示块池不足
    bool allocate_blocks(int32_t req_id,
                         base::BlockManager& block_manager,
                         int32_t block_size, int32_t max_gen_steps);

    // 调度参数（外部可调）
    int32_t max_active_requests_ = 64;
    int32_t max_prefill_tokens_per_step_ = 2048;

    // 公开（主循环直接遍历）
    std::unordered_map<int32_t, ScheduleRequest> active_requests_;

private:
    std::deque<ScheduleRequest> waiting_queue_;   // 等待队列
    int32_t next_req_id_ = 0;
};
```

### 1.2 调度器实现 (`kuiper/source/base/scheduler.cpp`)

```cpp
#include <base/scheduler.h>
#include <glog/logging.h>

// ────────────────────────────────────────────────
// submit: 新请求进入等待队列
//   1. 分配自增 req_id
//   2. 拷贝 prompt_tokens 和 max_gen_len
//   3. 推入 waiting_queue_
// ────────────────────────────────────────────────
int32_t Scheduler::submit(const std::vector<int32_t>& prompt_tokens,
                          int32_t max_gen_len) {
  ScheduleRequest req;
  req.req_id = next_req_id_++;
  req.prompt_tokens = prompt_tokens;
  req.prompt_len = static_cast<int32_t>(prompt_tokens.size());
  req.max_gen_len = max_gen_len;
  req.pos = 0;
  req.status = ReqStatus::kWaiting;
  waiting_queue_.push_back(req);
  return req.req_id;
}

// ────────────────────────────────────────────────
// schedule_step: 每步调度的核心决策函数
//
// 策略:
//   1. 收集 active_requests_ 中的所有请求
//      标记每个是 prefill (kPrefilling) 还是 decode
//   2. 计算空余 slot = max_active_requests_ - 当前活跃数
//   3. 从 waiting_queue_ 中取出请求准入 prefill
//      约束: 单步 prefill token 总数 ≤ max_prefill_tokens_per_step_
//   4. 返回 StepBatch（主循环据此组装 batch）
// ────────────────────────────────────────────────
Scheduler::StepBatch Scheduler::schedule_step() {
  StepBatch batch;

  // ── 阶段 1: 收集所有活跃请求 ──
  batch.active_indices.clear();
  batch.is_prefill.clear();
  int32_t cur_prefill_tokens = 0;

  for (auto& kv : active_requests_) {
    int32_t rid = kv.first;
    auto& req = kv.second;
    batch.active_indices.push_back(rid);

    if (req.status == ReqStatus::kPrefilling) {
      batch.is_prefill.push_back(true);
      // 统计剩余 prefill token 数
      cur_prefill_tokens += (req.prompt_len - req.pos);
    } else {
      batch.is_prefill.push_back(false);
    }
  }

  // ── 阶段 2: 从等待队列准入新请求 ──
  int32_t free_slots = max_active_requests_
                       - static_cast<int32_t>(active_requests_.size());

  while (free_slots > 0 && !waiting_queue_.empty()) {
    auto& next_req = waiting_queue_.front();

    // 单步 prefill token 上限：避免单步延迟尖峰
    if (cur_prefill_tokens + next_req.prompt_len
        > max_prefill_tokens_per_step_) {
      break;
    }

    // 准入：状态从 waiting → prefilling
    next_req.status = ReqStatus::kPrefilling;
    next_req.pos = 0;
    int32_t rid = next_req.req_id;
    active_requests_[rid] = next_req;
    waiting_queue_.pop_front();

    batch.active_indices.push_back(rid);
    batch.is_prefill.push_back(true);
    cur_prefill_tokens += next_req.prompt_len;
    free_slots--;
  }

  batch.total_batch_size =
      static_cast<int32_t>(batch.active_indices.size());
  return batch;
}

// ────────────────────────────────────────────────
// finish_request: 请求完成 → 回收物理块 → 移除
//   block_ids 由主循环在完成前从 block_table 收集
// ────────────────────────────────────────────────
void Scheduler::finish_request(int32_t req_id,
                               base::BlockManager& block_manager) {
  auto it = active_requests_.find(req_id);
  if (it == active_requests_.end()) return;

  auto& req = it->second;
  req.status = ReqStatus::kFinished;

  // 回收该请求的所有物理块，归还到全局块池
  if (!req.block_ids.empty()) {
    block_manager.free_blocks(req.block_ids);
    req.block_ids.clear();
  }

  active_requests_.erase(it);
}

// ────────────────────────────────────────────────
// 访问器
// ────────────────────────────────────────────────
ScheduleRequest& Scheduler::get_request(int32_t req_id) {
  auto it = active_requests_.find(req_id);
  CHECK(it != active_requests_.end())
      << "req_id=" << req_id << " not in active set";
  return it->second;
}

bool Scheduler::waiting_queue_empty() const {
  return waiting_queue_.empty();
}

int32_t Scheduler::active_count() const {
  return static_cast<int32_t>(active_requests_.size());
}

// ────────────────────────────────────────────────
// allocate_blocks: 为新准入 prefill 请求预分配物理块
//   计算 需要块数 = ceil((prompt_len + max_gen_len) / block_size)
//   检查块池余量，不足则拒绝准入
// ────────────────────────────────────────────────
bool Scheduler::allocate_blocks(int32_t req_id,
                                base::BlockManager& block_manager,
                                int32_t block_size,
                                int32_t max_gen_steps) {
  auto it = active_requests_.find(req_id);
  if (it == active_requests_.end()) return false;

  auto& req = it->second;
  int32_t total_tokens = req.prompt_len + max_gen_steps;
  int32_t blocks_needed =
      (total_tokens + block_size - 1) / block_size;

  if (block_manager.free_block_count() < blocks_needed) {
    return false;
  }

  req.block_ids.clear();
  for (int32_t i = 0; i < blocks_needed; ++i) {
    req.block_ids.push_back(block_manager.allocate_block());
  }
  return true;
}
```

---

## 2. Continuous Batching 主循环

### 2.1 block_table GPU 同步 (`demo/main.cpp`)

```cpp
// ────────────────────────────────────────────────
// 将 host 端的 block_table 全量同步到 GPU
//   关键：host 端 block_table 变更后必须调用此函数
//   否则 MHA kernel 查表时读到过期 block ID → KV 读错位
// ────────────────────────────────────────────────
static void sync_block_table_to_gpu(model::LLama2Model& model) {
  auto& bt = model.get_buffer(model::ModelBufferType::kBlockTable);
  int32_t* gpu_ptr = const_cast<int32_t*>(bt.ptr<int32_t>());
  size_t bytes =
      model.max_batch_size_ * model.max_blocks_per_req_ * sizeof(int32_t);
  cudaMemcpy(gpu_ptr,
             model.single_req_block_table_host_.data(),
             bytes, cudaMemcpyHostToDevice);
}
```

### 2.2 主函数 (`demo/main.cpp`)

```cpp
int32_t generate_batch_scheduled(
    model::LLama2Model& model,
    const std::vector<std::string>& sentences,
    int32_t max_gen_steps, bool need_output)
{
  // ──────────────────────────────────────────────
  // Step 0: 初始化调度器，提交所有初始和动态请求
  // ──────────────────────────────────────────────
  Scheduler scheduler;
  scheduler.max_active_requests_ = model.max_batch_size_;
  scheduler.max_prefill_tokens_per_step_ = 2048;

  // 提交初始 batch
  std::unordered_map<int32_t, int32_t> req_id_to_prompt_idx;
  std::vector<std::vector<int32_t>> generated_outputs;
  for (int32_t i = 0; i < (int32_t)sentences.size(); ++i) {
    auto tokens = model.encode(sentences[i]);
    int32_t rid = scheduler.submit(tokens, max_gen_steps);
    req_id_to_prompt_idx[rid] = i;
  }
  generated_outputs.resize(sentences.size());

  // 动态请求（推理中途才提交，演示 dynamic admission）
  struct DelayedReq { std::vector<int32_t> tokens; std::string prompt; };
  std::vector<DelayedReq> delayed;
  delayed.push_back({
    model.encode("The little bird wanted to fly high in the sky."),
    "The little bird wanted to fly high in the sky."});
  delayed.push_back({
    model.encode("It was a sunny day and the children played in the park."),
    "It was a sunny day and the children played in the park."});
  generated_outputs.resize(sentences.size() + delayed.size());

  // ──────────────────────────────────────────────
  // Step 1: 初始调度 — 准入所有初始请求
  // ──────────────────────────────────────────────
  scheduler.schedule_step();  // 将初始请求从 waiting → active

  // 重置全局 block_table 为全 -1（无请求占块）
  model.single_req_block_table_host_.assign(
      model.max_batch_size_ * model.max_blocks_per_req_, -1);
  sync_block_table_to_gpu(model);

  tensor::Tensor pos_tensor =
      model.get_buffer(model::ModelBufferType::kInputPos);
  int32_t total_steps = 0;
  bool has_active = true;

  // ──────────────────────────────────────────────
  // Step 2: CB 主循环
  //
  // 每步流程:
  //   schedule_step() → 获取本轮活跃请求列表
  //   Gather: 从每请求独立状态拼装 batch 输入
  //   Forward: 批量推理
  //   Scatter: 分发结果 → 采样 → 检测终止 → 回收
  // ──────────────────────────────────────────────
  while (has_active && total_steps < max_gen_steps * 2) {

    // ════════════════════════════════════════════════
    // Phase 0: 调度 — 决定本轮 batch 组成
    // ════════════════════════════════════════════════
    auto step_batch = scheduler.schedule_step();
    int32_t cur_batch = step_batch.total_batch_size;

    has_active = (!scheduler.waiting_queue_empty()
                  || scheduler.active_count() > 0);
    if (cur_batch == 0 && !has_active) break;
    if (cur_batch == 0) { total_steps++; continue; }

    // ════════════════════════════════════════════════
    // ★ 新请求准入：初始化 block_table 行（仅首次）
    //   条件: pos==0 确保只在准入后第一步骤执行
    //   如果用 block_ids.empty() 判断会每步都重置 → block 池爆炸
    // ════════════════════════════════════════════════
    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);
      if (req.status == ReqStatus::kPrefilling && req.pos == 0) {
        for (int32_t k = 0; k < model.max_blocks_per_req_; ++k) {
          model.single_req_block_table_host_[
              rid * model.max_blocks_per_req_ + k] = -1;
        }
      }
    }

    // ════════════════════════════════════════════════
    // Phase 1: Gather — 拼装 batch 输入
    //
    // 关键: 每个 batch slot 需要对应请求的 block_table 行
    //   block_table[s] ← block_table[rid]
    //   用 decode_saved 快照避免顺序依赖污染
    // ════════════════════════════════════════════════
    auto decode_saved = model.single_req_block_table_host_;
    std::vector<int32_t> cur_tokens(cur_batch);

    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);

      // ★ 每个请求设置自己的独立位置
      pos_tensor.index<int32_t>(s) = req.pos;

      // ★ Prefill/Decode 分离: token 来源不同
      if (req.status == ReqStatus::kPrefilling) {
        cur_tokens[s] = req.prompt_tokens[req.pos];
      } else {
        cur_tokens[s] = req.generated_ids.back();
      }

      // ★ 将 request row 映射到 batch slot row
      //   ensure_batch_blocks 按 slot 索引读取
      for (int32_t k = 0; k < model.max_blocks_per_req_; ++k) {
        model.single_req_block_table_host_[
            s * model.max_blocks_per_req_ + k] =
            decode_saved[rid * model.max_blocks_per_req_ + k];
      }
    }

    // ════════════════════════════════════════════════
    // Phase 2: Forward — 批量推理
    //   前: host→GPU 同步 block_table
    //   中: embedding + forward（所有请求共享一次 GPU 调用）
    //   后: writeback（batch slot → request row）
    // ════════════════════════════════════════════════
    sync_block_table_to_gpu(model);
    auto emb = model.embedding(cur_tokens);
    int dummy_next;
    model.forward(emb.input_embeddings, pos_tensor, dummy_next);
    cudaDeviceSynchronize();

    // ★ 写回: 将 batch slot 的 block_table 变更同步回 request row
    //   ensure_batch_blocks 可能在 forward 中分配了新块
    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      for (int32_t k = 0; k < model.max_blocks_per_req_; ++k) {
        decode_saved[rid * model.max_blocks_per_req_ + k] =
            model.single_req_block_table_host_[
                s * model.max_blocks_per_req_ + k];
      }
    }
    model.single_req_block_table_host_ = decode_saved;
    sync_block_table_to_gpu(model);

    // ════════════════════════════════════════════════
    // Phase 3: Scatter — 按需采样 + 状态更新
    //
    // Prefill 中间步: 不采样，仅 pos++（teacher forcing）
    // Prefill 最后一步: 采样，切换为 Decode
    // Decode 步: 采样，检测终止符
    // 终止: 收集 block_ids → finish_request（释放块）
    // ════════════════════════════════════════════════
    tensor::Tensor forward_output =
        model.get_buffer(model::ModelBufferType::kForwardOutput);
    int32_t vocab_size = std::abs(model.config_->vocab_size_);
    const float* device_logits = forward_output.ptr<float>();

    std::vector<int32_t> to_finish;
    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);

      // ── Prefill 中间步: teacher forcing，跳过采样 ──
      if (req.status == ReqStatus::kPrefilling
          && req.pos < req.prompt_len - 1) {
        req.pos++;
        continue;
      }

      // ── 采样: GPU argmax ──
      const float* b_logits = device_logits + s * vocab_size;
      int32_t next = static_cast<int32_t>(
          kernel::argmax_kernel_cu(b_logits, vocab_size, nullptr));
      req.generated_ids.push_back(next);
      req.pos++;

      // ── Prefill → Decode 切换 ──
      if (req.status == ReqStatus::kPrefilling) {
        req.status = ReqStatus::kDecoding;
      }

      // ── 终止检测 ──
      if (model.is_sentence_ending(next)
          || (int32_t)req.generated_ids.size() >= max_gen_steps) {
        to_finish.push_back(rid);
      }
    }

    // ★ 处理完成请求（延迟执行，避免迭代器失效）
    for (int32_t rid : to_finish) {
      auto& req = scheduler.get_request(rid);

      // 从 block_table 收集该请求的所有物理块 ID
      for (int32_t k = 0; k < model.max_blocks_per_req_; ++k) {
        int32_t blk = model.single_req_block_table_host_[
            rid * model.max_blocks_per_req_ + k];
        if (blk >= 0) req.block_ids.push_back(blk);
      }

      // 保存输出
      int32_t pidx = req_id_to_prompt_idx[rid];
      generated_outputs[pidx] = std::move(req.generated_ids);

      // ★ 释放物理块，归还到全局块池
      scheduler.finish_request(rid, model.block_manager_);
    }

    total_steps++;

    // ════════════════════════════════════════════════
    // 动态提交: 在指定步数将延迟请求注入等待队列
    //   schedule_step 在下一步自动准入（有空 slot 时）
    // ════════════════════════════════════════════════
    if (!delayed.empty() && total_steps == 60) {
      for (auto& dr : delayed) {
        int32_t rid = scheduler.submit(dr.tokens, max_gen_steps);
        req_id_to_prompt_idx[rid] =
            sentences.size() + (&dr - &delayed[0]);
        printf("[CB] step=%d: dynamic-submit req_id=%d \"%s\"\n",
               total_steps, rid, dr.prompt.c_str());
      }
      delayed.clear();
    }
  }

  // ── 输出结果 ──
  // ...
}
```

---

## 3. 核心改动点总结

### 3.1 Prefill/Decode 状态分离

| 维度 | 原始 Static Batching | Continuous Batching |
|------|---------------------|---------------------|
| **位置管理** | 全局 `pos`，所有请求共享 | 每请求独立 `req.pos` |
| **阶段标记** | 无（`pos < prompt_len` 隐式判断） | 显式 `ReqStatus` 枚举 |
| **Prefill 中间步** | 无区分，每步都走完整 forward | Teacher forcing：`pos++`，不采样 |
| **Prefill→Decode 切换** | 无显式切换 | `req.status = kDecoding` |

### 3.2 Block 生命周期管理

```
原始版本：请求启动时分配块，永远不释放
改造后：
  - 请求准入时：ensure_batch_blocks 按需分配
  - 请求完成时：从 block_table 行收集 block_ids
                → block_manager.free_blocks() 归还池
                → 新请求复用已释放的块
```

### 3.3 batch slot ↔ request row 映射

```
问题：ensure_batch_blocks 按 batch slot 索引读取 block_table
      但 block 逻辑上属于 request row（每个请求独占一行）

解决：每步 forward 前:
  1. 保存当前 block_table 到 decode_saved（快照）
  2. 将 request row 复制到 batch slot row
     host[slot] = decode_saved[rid]
  3. sync_host_to_gpu → forward → sync_gpu_to_host
  4. 将 batch slot row 写回 request row
     decode_saved[rid] = host[slot]
  5. host = decode_saved（恢复）

关键：步骤2必须从快照读取而非直接读host
      否则顺序映射时后一步会读到前一步覆盖的值
```

### 3.4 遍历安全性

```
问题：在 scatter 中遍历 active_requests_ 同时调用 finish_request
      finish_request 内部 erase map 元素 → 迭代器失效

解决：收集 to_finish 列表 → 循环结束后统一处理
```

### 3.5 新请求 block_table 重置时机

```
问题：新准入请求的 block_table 行可能有残留 block ID
     需重置为 -1，但每步都重置会导致块池爆炸

解决：仅在 req.pos == 0 时重置（准入后第一步）
     后续步骤由 gather/writeback 维护一致性
```

---

## 4. 量化对比

运行方式：

```bash
# 基线（PD 分离，8 请求）
./llama_infer model.bin tokenizer.model

# CB（8 初始 + 2 动态，10 请求）
./llama_infer model.bin tokenizer.model --scheduled
```

对比 METRICS 行：

| 指标 | 基线 (8 req) | CB (10 req) | 变化 |
|------|:-----------:|:-----------:|:----:|
| 请求数 | 8 | 10 | **+25%** |
| Prompt tokens | ~115 | ~136 | +18% |
| Generated tokens | ~1024 | ~1280 | **+25%** |
| 动态准入 | 不支持 | 2 个中途加入 | **新能力** |
| Block 回收 | 不回收 | 请求完成后释放 | **新能力** |
