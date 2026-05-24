#include <base/base.h>
#include <base/tick.h>
#include <base/scheduler.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include "../kuiper/source/op/kernels/cuda/argmax_kernel.cuh"

// ============================================================
// [P/D-分离] Per-request state — 每个请求独立跟踪位置和阶段
// ============================================================
struct ReqState {
  int32_t req_id;
  std::vector<int32_t> prompt_tokens;    // 完整 prompt token 列表
  std::vector<int32_t> generated_tokens; // 已生成的 token
  int32_t prompt_len;
  int32_t pos = 0;          // ★ 请求自己的当前处理位置（不再共享全局 pos）
  bool is_prefill = true;   // ★ true=预填充阶段, false=逐token解码阶段
  bool is_finished = false;
};

// ============================================================
// [P/D-分离] Prefill/Decode 分离版批量推理
//
// 与旧版核心区别：
//   1. pos 不再全局共享 → 每个 ReqState 独立维护
//   2. prefill 阶段走完所有 prompt token 之前不采样（teacher forcing）
//   3. 最后一个 prompt token 处理后采样一次，切换到 decode 阶段
//   4. decode 阶段每步采样一次，直到生成终止符
// ============================================================
int32_t generate_batch_pd(const model::LLama2Model& model,
                          const std::vector<std::string>& sentences,
                          int32_t max_gen_steps, bool need_output = false) {
  int32_t batch_size = sentences.size();
  std::vector<ReqState> reqs(batch_size);

  // ── 编码所有 prompt ──
  for (int32_t i = 0; i < batch_size; ++i) {
    reqs[i].req_id = i;
    reqs[i].prompt_tokens = model.encode(sentences[i]);
    reqs[i].prompt_len = reqs[i].prompt_tokens.size();
  }

  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  int32_t total_steps = 0;
  bool has_active = true;

  while (has_active && total_steps < max_gen_steps * 2) {
    has_active = false;

    // ════════════════════════════════════════════════════════
    // Phase 1: Gather — 根据 prefill/decode 状态选择输入 token
    // ════════════════════════════════════════════════════════
    std::vector<int32_t> cur_tokens(batch_size);
    for (int32_t b = 0; b < batch_size; ++b) {
      if (reqs[b].is_finished) {
        cur_tokens[b] = 0;            // padding token
        pos_tensor.index<int32_t>(b) = 0;
        continue;
      }
      has_active = true;

      // ★ 关键：每个请求设置自己的独立位置
      pos_tensor.index<int32_t>(b) = reqs[b].pos;

      if (reqs[b].is_prefill) {
        // Prefill: 来源 = prompt_tokens[当前pos]
        cur_tokens[b] = reqs[b].prompt_tokens[reqs[b].pos];
      } else {
        // Decode: 来源 = 上一轮生成的最后一个 token
        cur_tokens[b] = reqs[b].generated_tokens.back();
      }
    }
    if (!has_active) break;

    // ════════════════════════════════════════════════════════
    // Phase 2: Forward — prefill 和 decode 共用同一个 forward
    // ════════════════════════════════════════════════════════
    auto emb = model.embedding(cur_tokens);
    int dummy_next;
    auto st = model.forward(emb.input_embeddings, pos_tensor, dummy_next);
    if (!st) { LOG(FATAL) << "Forward Error: " << st.get_err_msg(); }
    cudaDeviceSynchronize();

    // ════════════════════════════════════════════════════════
    // Phase 3: Scatter — 只在需要的时候采样
    // ════════════════════════════════════════════════════════
    tensor::Tensor forward_output = model.get_buffer(
        model::ModelBufferType::kForwardOutput);
    int32_t vocab_size = std::abs(model.config_->vocab_size_);
    const float* device_logits = forward_output.ptr<float>();

    for (int32_t b = 0; b < batch_size; ++b) {
      if (reqs[b].is_finished) continue;

      if (reqs[b].is_prefill && reqs[b].pos < reqs[b].prompt_len - 1) {
        // ── Prefill 中间步：不采样，pos+1 继续 teacher forcing ──
        reqs[b].pos++;
        continue;
      }

      // ── Prefill 最后一步 / Decode 步：GPU argmax 采样 ──
      const float* b_logits = device_logits + b * vocab_size;
      int32_t next = static_cast<int32_t>(
          kernel::argmax_kernel_cu(b_logits, vocab_size, nullptr));
      reqs[b].generated_tokens.push_back(next);
      reqs[b].pos++;

      // Prefill → Decode 切换
      if (reqs[b].is_prefill) {
        reqs[b].is_prefill = false;
      }

      if (model.is_sentence_ending(next)) {
        reqs[b].is_finished = true;
      }
    }
    total_steps++;
  }

  // ── 输出 ──
  if (need_output) {
    for (int32_t b = 0; b < batch_size; ++b) {
      printf("\n==================================\n");
      printf("Prompt %d: %s\n", b, sentences[b].c_str());
      printf("Output %d: %s%s\n", b, sentences[b].c_str(),
             model.decode(reqs[b].generated_tokens).data());
      printf("==================================\n");
    }
    fflush(stdout);
  }

  int64_t prompt_total = 0, gen_total = 0;
  for (int32_t b = 0; b < batch_size; ++b) {
    prompt_total += reqs[b].prompt_len;
    gen_total += reqs[b].generated_tokens.size();
  }
  printf("METRICS prompt_tokens_total=%lld gen_tokens_total=%lld total_tokens_total=%lld\n",
         static_cast<long long>(prompt_total),
         static_cast<long long>(gen_total),
         static_cast<long long>(prompt_total + gen_total));
  fflush(stdout);

  return total_steps * batch_size;
}


// ============================================================
// [P/D-分离] Scheduler 驱动的 Continuous Batching 推理
//
// 与 generate_batch_pd 的核心区别:
//   1. 请求由 Scheduler 管理（可以中途动态提交新请求）
//   2. batch 组成每步可变（请求完成 → 释放 slot → 新请求准入）
//   3. block 通过 scheduler.allocate_blocks 预分配
// ============================================================
int32_t generate_batch_scheduled(model::LLama2Model& model,
                                 const std::vector<std::string>& sentences,
                                 int32_t max_gen_steps,
                                 bool need_output = false) {
  Scheduler scheduler;
  scheduler.max_active_requests_ = model.max_batch_size_;
  scheduler.max_prefill_tokens_per_step_ = 2048;

  // ── 提交所有请求到调度器 ──
  std::unordered_map<int32_t, int32_t> req_id_to_prompt_idx;
  for (int32_t i = 0; i < (int32_t)sentences.size(); ++i) {
    auto tokens = model.encode(sentences[i]);
    int32_t rid = scheduler.submit(tokens, max_gen_steps);
    req_id_to_prompt_idx[rid] = i;
  }

  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  auto& block_table_tensor = model.get_buffer(model::ModelBufferType::kBlockTable);
  int32_t* table_ptr = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());

  int32_t total_steps = 0;
  bool has_pending = true;

  while (has_pending && total_steps < max_gen_steps * 2) {
    // ════════════════════════════════════════════════════════
    // Phase 0: 调度 — 获取本轮活跃请求
    // ════════════════════════════════════════════════════════
    auto step_batch = scheduler.schedule_step();
    int32_t cur_batch = step_batch.total_batch_size;

    // 检查是否有等待中的请求或活跃请求
    has_pending = (!scheduler.waiting_queue_empty()
                   || scheduler.active_count() > 0);
    if (cur_batch == 0) {
      if (!has_pending) break;
      // 活跃请求数为 0 但有等待请求 → 下步重新调度
      total_steps++;
      continue;
    }

    // ── 为新准入的 prefill 请求分配 block ──
    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);
      if (req.status == ReqStatus::kPrefilling && req.block_ids.empty()) {
        bool ok = scheduler.allocate_blocks(
            rid, model.block_manager_, model.block_size_, max_gen_steps);
        if (!ok) {
          LOG(WARNING) << "Failed to allocate blocks for req_id=" << rid;
          // 退回等待队列（此处简化：直接标记完成）
          scheduler.finish_request(rid, model.block_manager_);
          continue;
        }
        // 把 block_ids 写入 block_table 的对应行
        int32_t row = s;
        for (size_t k = 0; k < req.block_ids.size(); ++k) {
          int32_t off = row * model.max_blocks_per_req_ + static_cast<int32_t>(k);
          table_ptr[off] = req.block_ids[k];
          model.single_req_block_table_host_[off] = req.block_ids[k];
        }
        // 初始化其余为 -1
        for (int32_t k = static_cast<int32_t>(req.block_ids.size());
             k < model.max_blocks_per_req_; ++k) {
          int32_t off = row * model.max_blocks_per_req_ + k;
          table_ptr[off] = -1;
          model.single_req_block_table_host_[off] = -1;
        }
      }
    }

    // ════════════════════════════════════════════════════════
    // Phase 1: Gather
    // ════════════════════════════════════════════════════════
    std::vector<int32_t> cur_tokens(cur_batch);
    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);

      pos_tensor.index<int32_t>(s) = req.pos;

      if (req.status == ReqStatus::kPrefilling) {
        cur_tokens[s] = req.prompt_tokens[req.pos];
      } else {
        cur_tokens[s] = req.generated_ids.back();
      }
    }

    // ════════════════════════════════════════════════════════
    // Phase 2: Forward
    // ════════════════════════════════════════════════════════
    auto emb = model.embedding(cur_tokens);
    int dummy_next;
    auto st = model.forward(emb.input_embeddings, pos_tensor, dummy_next);
    if (!st) { LOG(FATAL) << "Forward Error: " << st.get_err_msg(); }
    cudaDeviceSynchronize();

    // ════════════════════════════════════════════════════════
    // Phase 3: Scatter
    // ════════════════════════════════════════════════════════
    tensor::Tensor forward_output = model.get_buffer(
        model::ModelBufferType::kForwardOutput);
    int32_t vocab_size = std::abs(model.config_->vocab_size_);
    const float* device_logits = forward_output.ptr<float>();

    for (int32_t s = 0; s < cur_batch; ++s) {
      int32_t rid = step_batch.active_indices[s];
      auto& req = scheduler.get_request(rid);

      if (req.status == ReqStatus::kPrefilling
          && req.pos < req.prompt_len - 1) {
        req.pos++;
        continue;
      }

      const float* b_logits = device_logits + s * vocab_size;
      int32_t next = static_cast<int32_t>(
          kernel::argmax_kernel_cu(b_logits, vocab_size, nullptr));
      req.generated_ids.push_back(next);
      req.pos++;

      // Prefill → Decode 切换
      if (req.status == ReqStatus::kPrefilling) {
        req.status = ReqStatus::kDecoding;
      }

      // 终止或达上限
      if (model.is_sentence_ending(next)
          || static_cast<int32_t>(req.generated_ids.size()) >= max_gen_steps) {
        scheduler.finish_request(rid, model.block_manager_);
      }
    }

    total_steps++;
  }

  // ── 输出（按 prompt 索引查找结果） ──
  if (need_output) {
    for (int32_t i = 0; i < (int32_t)sentences.size(); ++i) {
      printf("\n==================================\n");
      printf("Prompt %d: %s\n", i, sentences[i].c_str());
      printf("==================================\n");
    }
    fflush(stdout);
  }

  return total_steps * model.max_batch_size_;
}


// ============================================================
// 旧版静态批量推理（保留，用于对比）
// ============================================================
int32_t generate_batch(const model::LLama2Model& model, const std::vector<std::string>& sentences,
                       int32_t total_steps, bool need_output = false) {
  int32_t batch_size = sentences.size();
  std::vector<std::vector<int32_t>> tokens_batch;
  std::vector<int32_t> prompt_lens;
  
  for (const auto& s : sentences) {
      auto tokens = model.encode(s);
      tokens_batch.push_back(tokens);
      prompt_lens.push_back(tokens.size());
  }

  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  
  std::vector<std::vector<int32_t>> words_batch(batch_size);
  std::vector<int32_t> next_tokens(batch_size, -1);
  std::vector<bool> is_finished(batch_size, false);
  
  int32_t pos = 0;
  
  while (pos < total_steps) {
      bool all_finished = true;
      for (bool f : is_finished) {
          if (!f) all_finished = false;
      }
      if (all_finished) break;

      std::vector<int32_t> current_tokens;
      for (int b = 0; b < batch_size; ++b) {
          pos_tensor.index<int32_t>(b) = pos;
          if (pos < prompt_lens[b]) {
              current_tokens.push_back(tokens_batch[b][pos]);
          } else {
              current_tokens.push_back(next_tokens[b]);
          }
      }

      auto embedding_output = model.embedding(current_tokens);
      auto input = embedding_output.input_embeddings;
      
      int dummy_next;
      auto st = model.forward(input, pos_tensor, dummy_next);
      if (!st) { LOG(FATAL) << "Forward Error: " << st.get_err_msg(); }
      cudaError_t err = cudaDeviceSynchronize();
      if (err != cudaSuccess) { LOG(FATAL) << "CUDA Error after forward: " << cudaGetErrorString(err); }

      tensor::Tensor forward_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
      int32_t vocab_size = std::abs(model.config_->vocab_size_);
      const float* device_logits = forward_output.ptr<float>();
      
    //   std::vector<float> host_logits(batch_size * vocab_size);
    //   cudaMemcpy(host_logits.data(), forward_output.ptr<float>(), batch_size * vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

      for (int b = 0; b < batch_size; ++b) {
          if (is_finished[b]) continue;

        //   float* b_logits = host_logits.data() + b * vocab_size;
        //   int32_t next = std::distance(b_logits, std::max_element(b_logits, b_logits + vocab_size));
          int32_t next = -1;
          
          if (pos >= prompt_lens[b] - 1) {
            const float* b_logits = device_logits + b * vocab_size;
            // 最小改法：每行在GPU做argmax，只回拷一个index
            next = static_cast<int32_t>(kernel::argmax_kernel_cu(b_logits, vocab_size, nullptr));
            next_tokens[b] = next;
            words_batch[b].push_back(next);
            if (model.is_sentence_ending(next)) {
                is_finished[b] = true;
            }
          }
      }

      pos += 1;
  }

  if (need_output) {
      for (int b = 0; b < batch_size; ++b) {
          printf("\n==================================\n");
          printf("Prompt %d: %s\n", b, sentences[b].c_str());
          printf("Output %d: %s%s\n", b, sentences[b].c_str(), model.decode(words_batch[b]).data());
          printf("==================================\n");
      }
      fflush(stdout);
  }

  int64_t prompt_tokens_total = 0;
  int64_t gen_tokens_total = 0;
  for (int b = 0; b < batch_size; ++b) {
      prompt_tokens_total += prompt_lens[b];
      gen_tokens_total += static_cast<int64_t>(words_batch[b].size());
  }
  int64_t total_tokens_total = prompt_tokens_total + gen_tokens_total;
  printf("METRICS prompt_tokens_total=%lld gen_tokens_total=%lld total_tokens_total=%lld\n",
         static_cast<long long>(prompt_tokens_total),
         static_cast<long long>(gen_tokens_total),
         static_cast<long long>(total_tokens_total));
  fflush(stdout);

  return pos * batch_size; 
}

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path [--quant]";
    return -1;
  }
  bool use_quant = false;
  if (argc == 4) {
    use_quant = (std::string(argv[3]) == "--quant");
    if (!use_quant) {
      LOG(INFO) << "Unknown option: " << argv[3];
      LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path [--quant]";
      return -1;
    }
  }
  const char* checkpoint_path = argv[1];  
  const char* tokenizer_path = argv[2];

  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "/home/wyk/KuiperLLama_new/logs";
  FLAGS_logtostderr = false;
  FLAGS_alsologtostderr = false;

  // 1. 初始化模型类
  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, use_quant);

  // 2. 使用不等长 prompt 体现 P/D 分离价值
  //    不同 prompt 长度不同 → prefill 步数不同 → 各自独立切换 decode
  std::vector<std::string> sentences;
  sentences.push_back("Hello, who are you?");
  sentences.push_back("Once upon a time, there was a little girl who lived in a small village.");
  sentences.push_back("The capital of France is Paris. Paris is a beautiful city known for");
  sentences.push_back("What is the meaning of life? The answer to this question is");
  sentences.push_back("Tell me a short story about a brave knight.");
  sentences.push_back("Prompt number 5 is a very nice prompt.");
  sentences.push_back("In this paper, we propose a novel approach to");
  sentences.push_back("The quick brown fox jumps over the lazy dog. This sentence contains every");

  printf("===== Prefill/Decode Separated Batch Inference =====\n");
  for (int i = 0; i < (int)sentences.size(); ++i) {
    printf("Prompt %d (len=%d): %s\n", i,
           (int)model.encode(sentences[i]).size(), sentences[i].c_str());
  }
  fflush(stdout);

  // 3. 核心配置：最大并发数 = prompt 数量
  model.max_batch_size_ = sentences.size();

  // 4. 调用 init 申请底层 Paged KV Cache 和 Block Table
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed: " << init_status.get_err_code();
  }

  auto start = std::chrono::steady_clock::now();

  // 5. ★ Scheduler 驱动的 P/D 分离推理
  int total_generated = generate_batch_scheduled(model, sentences, 128, true);
  
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(total_generated) / duration);
  fflush(stdout);
  return 0;
}
