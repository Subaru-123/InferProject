// #include <base/base.h>
// #include <base/tick.h>
// #include <glog/logging.h>
// #include "model/llama3.h"
// int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
//                  bool need_output = false) {
//   auto tokens = model.encode(sentence);
//   int32_t prompt_len = tokens.size();
//   LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

//   int32_t pos = 0;
//   int32_t next = -1;
//   bool is_prompt = true;
//   const auto& prompt_embedding = model.embedding(tokens);
//   tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

//   std::vector<int32_t> words;
//   while (pos < total_steps) {
//     pos_tensor.index<int32_t>(0) = pos;
//     if (pos < prompt_len - 1) {
//       tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
//       model.predict(input, pos_tensor, is_prompt, next);
//     } else {
//       is_prompt = false;
//       tokens = std::vector<int32_t>{next};
//       const auto& token_embedding = model.embedding(tokens);
//       tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
//       model.predict(input, pos_tensor, is_prompt, next);
//     }
//     if (model.is_sentence_ending(next)) {
//       break;
//     }
//     if (is_prompt) {
//       next = tokens.at(pos + 1);
//       words.push_back(next);
//     } else {
//       words.push_back(next);
//     }

//     pos += 1;
//   }
//   if (need_output) {
//     printf("%s ", model.decode(words).data());
//     fflush(stdout);
//   }
//   return std::min(pos, total_steps);
// }


// int main(int argc, char* argv[]) {
//   if (argc != 3) {
//     LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
//     return -1;
//   }
//   const char* checkpoint_path = argv[1];  // e.g. out/model.bin
//   const char* tokenizer_path = argv[2];

//   model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
//     checkpoint_path, false);
//   auto init_status = model.init(base::DeviceType::kDeviceCUDA);
//   if (!init_status) {
//     LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
//   }
//   const std::string& sentence = "hello";

//   auto start = std::chrono::steady_clock::now();
//   printf("Generating...\n");
//   fflush(stdout);
//   int steps = generate(model, sentence, 128, true);
//   auto end = std::chrono::steady_clock::now();
//   auto duration = std::chrono::duration<double>(end - start).count();
//   printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
//   fflush(stdout);
//   return 0;
// }


#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama3.h"
#include <vector>
#include <string>
#include <chrono>

// 定义每个并发请求的独立状态, 去除独立的 block_table
struct ReqState {
  int req_id;
  std::vector<int32_t> tokens;
  int32_t prompt_len;
  int32_t pos = 0;
  // int32_t next = -1;
  bool is_prompt = true;
  bool finished = false;
  // std::vector<int32_t> block_table; // 核心：独占的物理显存映射表
};

int32_t generate_batched(model::LLama2Model& model, const std::string& sentence, 
                            int batch_size, int total_steps) {
  // 1. 初始化并发请求池
  std::vector<ReqState> reqs(batch_size);
  std::vector<int32_t> prompt_tokens = model.encode(sentence);
  
  // 核心：由外部调度器接管全局 2D 物理映射表 [batch_size, max_blocks]
  int32_t max_blocks = (total_steps + model.block_size_ - 1) / model.block_size_;
  std::vector<int32_t> block_table_2d(batch_size * max_blocks, 0);

  for (int i = 0; i < batch_size; ++i) {
    reqs[i].req_id = i;
    reqs[i].tokens = model.encode(sentence);
    reqs[i].prompt_len = reqs[i].tokens.size();

    //LOG_IF(FATAL, reqs[i].tokens.empty()) << "The tokens is empty.";
    // 提前为每个请求在全局 2D 表中分配足额的物理块 
    // (工业界这里是按需分配，为了极限压测我们一次性分配)
    for (int j = 0; j < max_blocks; ++j) {
      block_table_2d[i * max_blocks + j] = model.block_manager_.allocate_block();
    }
  }

  // 2. 准备 Batched 张量 (直接向框架索要 Buffer 并 Reshape)
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
  pos_tensor.reshape({batch_size}); // 扩充为 [batch_size]

  tensor::Tensor input_embeddings = model.get_buffer(model::ModelBufferType::kInputEmbeddings);
  input_embeddings.reshape({batch_size, model.config_->dim_}); // 扩充为 [batch_size, dim]

  tensor::Tensor block_table_tensor = model.get_buffer(model::ModelBufferType::kBlockTable);
  block_table_tensor.reshape({batch_size * max_blocks}); // 扩充为 [batch_size * max_blocks]

  int32_t total_generated = 0;
  bool has_active_request = true;

  // 3. Batched 推理主循环
  while (has_active_request) {
    has_active_request = false;

    // ==========================================
    //  Step 1: Gather (动态拼接 Batch)
    // ==========================================
    std::vector<int32_t> batch_tokens;
    for (int i = 0; i < batch_size; ++i) {
      if (reqs[i].pos >= total_steps) reqs[i].finished = true;
      // if (reqs[i].finished) continue;
      if (reqs[i].finished) {
        batch_tokens.push_back(0); // 填充无效的 padding token，保持矩阵形状完美
      } else {
        has_active_request = true;
        int32_t current_token = reqs[i].is_prompt ? reqs[i].tokens[reqs[i].pos] : reqs[i].tokens.back();
        batch_tokens.push_back(current_token);
      }
      pos_tensor.index<int32_t>(i) = reqs[i].pos;

      // // 提取当前 token 的 Embedding，拼接到大矩阵 input_embeddings 的对应行中
      // int32_t current_token = reqs[i].is_prompt ? reqs[i].tokens[reqs[i].pos] : reqs[i].tokens.back();
      // std::vector<int32_t> single_token = {current_token};
      // const auto& emb = model.embedding(single_token);
      
      // // 暴力但高效的内存拷贝拼接
      // memcpy(const_cast<float*>(input_embeddings.ptr<float>()) + i * model.config_->dim_,
      //        emb.input_embeddings.ptr<float>(), 
      //        model.config_->dim_ * sizeof(float));
    }

    if (!has_active_request) break;

    // 同步完整的 2D 物理块映射表到 GPU
    cudaMemcpyAsync(const_cast<int32_t*>(block_table_tensor.ptr<int32_t>()),
                    block_table_2d.data(), 
                    block_table_2d.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice, model.cuda_config_->stream);

    // ==========================================
    //  Step 2: Forward (Batched 一波流核爆轰炸)
    // ==========================================
    // 核心修复：直接将整批 Token 传给 embedding，由底层 GPU Kernel 并行生成矩阵，彻底消灭 memcpy！
    model.embedding(batch_tokens);
    
    // 注意：这里不再调用 predict，而是直接调用底层的 forward！
    // 所有的算子都会被喂入 [batch_size, dim] 的大矩阵！
    int dummy_next = -1; 
    // 直接获取已经被完美填充的连续 GPU 内存
    //tensor::Tensor batched_input_embeddings = model.get_buffer(model::ModelBufferType::kInputEmbeddings);
    // 【关键】一定要把拷贝出来的局部变量 reshape 一下，再传进去！
    //batched_input_embeddings.reshape({batch_size, model.config_->dim_});

    // --- 核心救命代码：构造零拷贝 View，彻底消灭所有显存重新分配和泄漏！ ---
    tensor::Tensor batched_input_embeddings(
        base::DataType::kDataTypeFp32, 
        batch_size * model.config_->dim_, 
        false, 
        nullptr, 
        const_cast<void*>(model.get_buffer(model::ModelBufferType::kInputEmbeddings).ptr<void>())
    );
    batched_input_embeddings.reshape({batch_size, model.config_->dim_});
    batched_input_embeddings.set_device_type(base::DeviceType::kDeviceCUDA);
    // 在 main.cpp 的 model.forward 调用前强制重置一次 Shape
    //input_embeddings.reshape({batch_size, model.config_->dim_});
    model.forward(batched_input_embeddings, pos_tensor, dummy_next);
    

    // ==========================================
    //  Step 3: Scatter (分发与状态更新)
    // ==========================================
    // 强制同步以获取结果
    cudaStreamSynchronize(model.cuda_config_->stream);
    
    // 从输出矩阵中抽取每个请求的 Logits 并进行采样
    tensor::Tensor forward_output = model.get_buffer(model::ModelBufferType::kForwardOutput);
    const float* batched_logits = forward_output.ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
      if (reqs[i].finished) continue;

      int32_t next_token = -1;
      if (reqs[i].is_prompt) {
        next_token = reqs[i].tokens.at(reqs[i].pos + 1);
      } else {
        // 从 [batch_size, vocab_size] 矩阵中定位当前请求的那一行进行采样
        // --- 核心修复：严防 vocab_size 为负数的幽灵指针偏移 ---
        int32_t vocab_size = std::abs(model.config_->vocab_size_);
        const float* req_logits = batched_logits + i * vocab_size;
        next_token = static_cast<int32_t>(model.sampler_->sample(
            req_logits, vocab_size, model.cuda_config_->stream));
        reqs[i].tokens.push_back(next_token);
      }

      if (model.is_sentence_ending(next_token)) {
        reqs[i].finished = true;
      } else {
        reqs[i].pos += 1;
        total_generated += 1;
      }

      if (reqs[i].pos >= reqs[i].prompt_len - 1) {
        reqs[i].is_prompt = false;
      }
    }
  }
      // model.set_current_block_table(reqs[i].block_table);
      // pos_tensor.index<int32_t>(0) = reqs[i].pos;

      // 执行模型前向传播
  //     if (req.pos < req.prompt_len - 1) {
  //       const auto& prompt_embedding = model.embedding(req.tokens);
  //       tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, req.is_prompt);
  //       model.predict(input, pos_tensor, req.is_prompt, req.next);
  //     } else {
  //       req.is_prompt = false;
  //       std::vector<int32_t> single_token = {req.next};
  //       const auto& token_embedding = model.embedding(single_token);
  //       tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, req.is_prompt);
  //       model.predict(input, pos_tensor, req.is_prompt, req.next);
  //     }

  //     // ==========================================
  //     //  Context Switch Out: 保存分配的物理表
  //     // ==========================================
  //     req.block_table = model.get_current_block_table();

  //     // 状态更新
  //     if (model.is_sentence_ending(req.next)) {
  //       req.finished = true;
  //       // 注意：生产环境中这里应调用 block_manager_.free_blocks 释放显存，此处仅做极限容量压测
  //     } else {
  //       if (req.is_prompt) {
  //         req.next = req.tokens.at(req.pos + 1);
  //       } else {
  //         req.tokens.push_back(req.next); // 记录生成的词
  //       }
  //       req.pos += 1;
  //       total_generated += 1;
  //     }
  //   }
  // }

  // 打印第一个和最后一个请求的生成结果作为验证
  printf("\n[请求 0 生成结果]: %s \n", model.decode(reqs[0].tokens).data());
  printf("\n[请求 %d 生成结果]: %s \n", batch_size - 1, model.decode(reqs.back().tokens).data());
  // printf("\n[请求 %d 生成结果]: %s \n", num_requests - 1, model.decode(reqs.back().tokens).data());
  return total_generated;
}

int main(int argc, char* argv[]) {
  // 增加了一个可选参数：并发请求数 batch_size
  if (argc < 3) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path [batch_size]";
    return -1;
  }
  const char* checkpoint_path = argv[1];
  const char* tokenizer_path = argv[2];
  int batch_size = (argc == 4) ? std::stoi(argv[3]) : 1; // 默认 1 个并发

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  
  const std::string& sentence = "hello, who was a little girl. She was three years old";

  printf("========== Paged Attention 极限并发压测 ==========\n");
  printf("当前压测并发数 (Batch Size): %d\n", batch_size);
  printf("引擎状态：时间片轮转已销毁，开启 Batched GEMM 矩阵聚变\n");
  printf("生成中，请稍候...\n");
  fflush(stdout);

  auto start = std::chrono::steady_clock::now();
  
  // 执行多并发生成，每个请求最大长度 128
  int total_tokens = generate_batched(model, sentence, batch_size, 128);
  
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  
  printf("==================================================\n");
  printf("总耗时: %lf 秒\n", duration);
  printf("总生成 Token 数: %d\n", total_tokens);
  printf("系统总吞吐量: %lf Tokens/s\n", static_cast<double>(total_tokens) / duration);
  printf("==================================================\n");
  fflush(stdout);
  return 0;
}