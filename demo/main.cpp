#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include <vector>
#include <string>
#include <chrono>
#include "../kuiper/source/op/kernels/cuda/argmax_kernel.cuh"

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

  // 2. 准备你想并发测试的任意多条句子 (例如 6 条)
    std::vector<std::string> sentences;
  for (int i = 0; i < 20; ++i) {
      sentences.push_back("Prompt number " + std::to_string(i) + " is a very nice prompt.");
  }

  // 3. 【核心配置】：在模型分配显存 (init) 前，告诉它我们要跑的最大并发数！
  model.max_batch_size_ = sentences.size(); 

  // 4. 调用 init 申请底层 Paged KV Cache、Block Table 和 QKV 缓存
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed: " << init_status.get_err_code();
  }

  auto start = std::chrono::steady_clock::now();
  printf("Generating Batch (Size %d)...\n", (int)sentences.size());
  for(int i=0;i<sentences.size();i++) {
      printf("Prompt %d len: %d\n", i, (int)model.encode(sentences[i]).size());
  }
  fflush(stdout);
  
  // 5. 启动 Continuous Batching 推理
  int total_generated = generate_batch(model, sentences, 128, true);
  
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(total_generated) / duration);
  fflush(stdout);
  return 0;
}
