#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include <vector>
#include <random>
#include "base/block_manager.h"
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  std::vector<int32_t> get_current_block_table() const {return current_block_table_;}

  void set_current_block_table(const std::vector<int32_t>& table) {
    const_cast<LLama2Model*>(this)->current_block_table_ = table;
  }

  void ensure_batch_blocks(const tensor::Tensor& pos_tensor) const;

  struct SamplingConfig {
    float repetition_penalty = 1.0f;
    int repetition_window = 64;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    bool do_sample = true;
  };

  void set_sampling_config(const SamplingConfig& config);
  void set_sampling_history(const std::vector<int32_t>& history) const;
  void reset_sampling_history() const;
  void append_sampling_token(int32_t token) const;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

 public:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
  std::vector<int32_t> current_block_table_;
  mutable base::BlockManager block_manager_;
  int32_t block_size_ = 16;
  int32_t max_batch_size_ = 1;
  int32_t max_blocks_per_req_ = 1;
  mutable std::vector<int32_t> single_req_block_table_host_;
  mutable tensor::Tensor rets_logit_;
  SamplingConfig sampling_config_{};
  mutable std::vector<int32_t> sampling_history_;
  mutable std::mt19937 sampling_rng_{std::random_device{}()};
};

}  // namespace model

#endif