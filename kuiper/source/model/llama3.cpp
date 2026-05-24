#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <array>
#include <limits>
#include <cstring>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/kvcache_kernel.cuh"
#include "base/tick.h"
namespace model {

void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) { add_layer_->set_cuda_config(config); add_layer_->to_cuda(); }
  if (rope_layer_) { rope_layer_->set_cuda_config(config); rope_layer_->to_cuda(); }
  if (swiglu_layer_) { swiglu_layer_->set_cuda_config(config); swiglu_layer_->to_cuda(); }
  if (cls_layer_) { cls_layer_->set_cuda_config(config); cls_layer_->to_cuda(); }
  if (embedding_layer_) { embedding_layer_->set_cuda_config(config); embedding_layer_->to_cuda(); }
  if (mha_layer_) { mha_layer_->set_cuda_config(config); mha_layer_->to_cuda(); }
  for (auto& weight_layer : wq_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : wk_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : wv_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : wo_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : w1_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : w2_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& weight_layer : w3_layers_) { if (weight_layer) { weight_layer->set_cuda_config(config); weight_layer->to_cuda(); } }
  for (auto& rms_norm_layer : rmsnorm_layers_) { if (rms_norm_layer) { rms_norm_layer->to_cuda(); rms_norm_layer->set_cuda_config(config); } }
}

void LLama2Model::set_sampling_config(const SamplingConfig& config) {
  sampling_config_ = config;
}

void LLama2Model::set_sampling_history(const std::vector<int32_t>& history) const {
  sampling_history_ = history;
}

void LLama2Model::reset_sampling_history() const {
  sampling_history_.clear();
}

void LLama2Model::append_sampling_token(int32_t token) const {
  sampling_history_.push_back(token);
}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) return error::PathNotValid(token_path_);
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) return error::InternalError("Unsupported int8 quant");
  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    int device_id = 0;
    cudaError_t set_device_err = cudaSetDevice(device_id);
    if (set_device_err != cudaSuccess) {
      return error::InternalError("cudaSetDevice failed: " + std::string(cudaGetErrorString(set_device_err)));
    }
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaDeviceProp prop{};
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    LOG(INFO) << "LLama2Model initialized on CUDA device " << device_id
              << " (" << prop.name << ", SM " << prop.major << '.' << prop.minor << ")";
  } else {
    LOG(INFO) << "LLama2Model initialized on CPU";
  }
  Status read_status = gen_model_from_file();
  if (!read_status) return read_status;
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_, get_buffer(ModelBufferType::kSinCache).ptr<float>(), get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_, get_buffer(ModelBufferType::kSinCache), get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }
  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor, int& next) const {
  int32_t batch_size = pos_tensor.size(); 
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  
  tensor::Tensor pos_gpu(base::DataType::kDataTypeInt32, batch_size, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kPosGPU).ptr<void>()));
  pos_gpu.reshape({batch_size});
  pos_gpu.set_device_type(device_type_);
  
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cudaMemcpyAsync(pos_gpu.ptr<int32_t>(), pos_tensor.ptr<int32_t>(), 
                    batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  } 

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    // removed
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_, config_->head_size_);
  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  llama_layers_->swiglu_layer_ = std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void LLama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  // cls_layer->set_group_size(group_size_);
  // if (config_->is_shared_weight_) {
  //   cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  // } else {
  //   cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  //   pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  // }
  std::shared_ptr<op::MatmulLayer> cls_layer;
  if (config_->is_shared_weight_) {
    // shared 情况下，version3 导出不会写量化 lm_head；这里直接使用 fp32 embedding 权重
    cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, false);
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  } else {
    cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
    cls_layer->set_group_size(group_size_);
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void LLama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_}, weight_embedding, cpu_device_type);

  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq); pos += dim * dim;
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk); pos += config_->kv_dim_ * dim;
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv); pos += config_->kv_dim_ * dim;
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo); pos += dim * dim;
  }
  pos += config_->layer_num_ * dim;
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1); pos += dim * hidden_dim;
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2); pos += dim * hidden_dim;
  }
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3); pos += dim * hidden_dim;
  }
  pos += dim; pos += config_->seq_len_ * config_->head_size_;
  llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
  }
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    rms_norm_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos), cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer); rmsnorm_pos += config_->dim_;
  }
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    rms_norm_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos), cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer); rmsnorm_pos += config_->dim_;
  }
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  std::shared_ptr<op::RmsNormLayer> rms_final_layer = std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
  rms_final_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos), cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

void LLama2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }
  std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 默认采用单请求配置，避免一次性预分配 1024 并发造成显存峰值
  // max_batch_size_ is now configured before init() by the user
  max_blocks_per_req_ = std::max(1, (config_->seq_len_ + block_size_ - 1) / block_size_);

  const int32_t total_physical_blocks = max_batch_size_ * max_blocks_per_req_;
  block_manager_.init(total_physical_blocks);

  if (llama_layers_ && llama_layers_->mha_layer_) {
    auto mha_layer = std::dynamic_pointer_cast<op::MultiHeadAttention>(llama_layers_->mha_layer_);
    if (mha_layer) {
      mha_layer->set_total_blocks(block_manager_.total_blocks());
    }
  }

  tensor::Tensor block_table(base::DataType::kDataTypeInt32,
                             max_batch_size_ * max_blocks_per_req_, true, alloc);
  block_table.reshape({max_batch_size_, max_blocks_per_req_});
  CHECK(insert_buffer(ModelBufferType::kBlockTable, block_table));

  single_req_block_table_host_.assign(max_batch_size_ * max_blocks_per_req_, -1);

  tensor::Tensor key_temp(base::DataType::kDataTypeFp32, max_batch_size_ * config_->kv_dim_, true, alloc);
  tensor::Tensor val_temp(base::DataType::kDataTypeFp32, max_batch_size_ * config_->kv_dim_, true, alloc);
  tensor::Tensor pos_gpu(base::DataType::kDataTypeInt32, max_batch_size_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyTemp, key_temp));
  CHECK(insert_buffer(ModelBufferType::kValueTemp, val_temp));
  CHECK(insert_buffer(ModelBufferType::kPosGPU, pos_gpu));

  int32_t max_input_len = std::max(max_batch_size_, config_->seq_len_);
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, max_input_len, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, max_input_len, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, max_batch_size_ * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, max_batch_size_ * config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, max_batch_size_ * config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, total_physical_blocks * block_size_, config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, total_physical_blocks * block_size_, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  tensor::Tensor query(base::DataType::kDataTypeFp32, max_batch_size_ * config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, max_batch_size_, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  tensor::Tensor attn(base::DataType::kDataTypeFp32, max_batch_size_ * config_->head_num_ * config_->seq_len_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));

  int32_t vocab_size = std::abs(config_->vocab_size_);
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, max_batch_size_ * vocab_size, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, max_batch_size_ * vocab_size, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

void LLama2Model::ensure_batch_blocks(const tensor::Tensor& pos_tensor) const {
  if (single_req_block_table_host_.empty()) {
    single_req_block_table_host_.assign(max_batch_size_ * max_blocks_per_req_, -1);
  }
  auto& block_table_tensor = get_buffer(ModelBufferType::kBlockTable);
  int32_t* table_ptr = const_cast<int32_t*>(block_table_tensor.ptr<int32_t>());
  int32_t batch_size = pos_tensor.size();

  for (int b = 0; b < batch_size; ++b) {
    int32_t pos = pos_tensor.index<int32_t>(b);
    int32_t logical_block_idx = pos / block_size_;
    if (logical_block_idx < 0 || logical_block_idx >= max_blocks_per_req_) continue;

    int32_t table_offset = b * max_blocks_per_req_ + logical_block_idx;
    if (single_req_block_table_host_[table_offset] != -1) continue;

    int32_t block_id = block_manager_.allocate_block();
    CHECK_GE(block_id, 0) << "Block pool exhausted";
    single_req_block_table_host_[table_offset] = block_id;

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      cudaMemcpy(table_ptr + table_offset, &block_id, sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
      table_ptr[table_offset] = block_id;
    }
  }
}

base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) llama_layers_ = std::make_unique<LLama2Layers>();
  if (!is_quant_model_) create_param_layers();
  else create_param_quant_layers();
  create_nonparam_layers();
  return error::Success();
}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  int32_t batch_size = tokens.size();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, batch_size, false, alloc_cpu, const_cast<void*>(get_buffer(ModelBufferType::kInputTokens).ptr<void>()));
  input_tokens.reshape({batch_size});
  input_tokens.set_device_type(base::DeviceType::kDeviceCPU);

  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kInputEmbeddings).ptr<void>()));
  input_embeddings.reshape({batch_size, config_->dim_});
  input_embeddings.set_device_type(device_type_);

  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  // 必须传入 alloc_cpu 并开启 true，防止底层空指针越界
  auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, batch_size, true, alloc_cpu);
  STATUS_CHECK(llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  int32_t batch_size = input.size() / config_->dim_; 
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  
  tensor::Tensor rmsnorm_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kOutputRMSNorm).ptr<void>()));
  rmsnorm_output.reshape({batch_size, config_->dim_}); 
  rmsnorm_output.set_device_type(device_type_);

  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);

  // --- 核心修复 1：将大矩阵切片，彻底阻断 RMSNorm 跨批次求和带来的污染 ---
  for (int i = 0; i < batch_size; ++i) {
    tensor::Tensor in_view(input.data_type(), config_->dim_, false, alloc_cu, 
                           const_cast<float*>(input.ptr<float>()) + i * config_->dim_);
    tensor::Tensor out_view(rmsnorm_output.data_type(), config_->dim_, false, alloc_cu, 
                            rmsnorm_output.ptr<float>() + i * config_->dim_);
    in_view.set_device_type(device_type_);
    out_view.set_device_type(device_type_);
    STATUS_CHECK(rmsnorm_layer->forward(in_view, out_view));
  }
  // STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  int32_t batch_size = pos_tensor.size(); 
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 1. 从预分配池 kQuery 中拿出 Buffer，绑定给当前的临时张量 query
  tensor::Tensor query(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kQuery).ptr<void>()));
  // 2. 推理时仅改变 Shape，将其调整为当前 batch 实际需要的形状
  query.reshape({batch_size, config_->dim_});
  // 之后将这块复用的内存传入算子进行计算
  query.set_device_type(device_type_);

  tensor::Tensor pos_gpu(base::DataType::kDataTypeInt32, batch_size, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kPosGPU).ptr<void>()));
  pos_gpu.reshape({batch_size});
  pos_gpu.set_device_type(device_type_);
  
  tensor::Tensor key(base::DataType::kDataTypeFp32, batch_size * config_->kv_dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kKeyTemp).ptr<void>()));
  key.reshape({batch_size, config_->kv_dim_});
  key.set_device_type(device_type_);

  tensor::Tensor val(base::DataType::kDataTypeFp32, batch_size * config_->kv_dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kValueTemp).ptr<void>()));
  val.reshape({batch_size, config_->kv_dim_});
  val.set_device_type(device_type_);

  tensor::Tensor rmsnorm_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kOutputRMSNorm).ptr<void>()));
  rmsnorm_output.reshape({batch_size, config_->dim_});
  rmsnorm_output.set_device_type(device_type_);

  ensure_batch_blocks(pos_tensor);

#ifdef DEBUG_SINGLE_REQ
  if (max_batch_size_ == 1 && batch_size == 1 && device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor block_table = get_buffer(ModelBufferType::kBlockTable);
    std::vector<int32_t> gpu_table(max_blocks_per_req_, -1);
    size_t bytes = static_cast<size_t>(max_blocks_per_req_) * sizeof(int32_t);
    cudaError_t snapshot_status = cudaMemcpy(gpu_table.data(), block_table.ptr<int32_t>(), bytes, cudaMemcpyDeviceToHost);
    CHECK_EQ(snapshot_status, cudaSuccess) << "Failed to snapshot GPU block table inside attention_mha";

    std::ostringstream gpu_dump;
    gpu_dump << "[single-request][attention_mha] block table head:";
    int gpu_limit = max_blocks_per_req_ < 8 ? max_blocks_per_req_ : 8;
    for (int i = 0; i < gpu_limit; ++i) {
      gpu_dump << " [" << i << "]=" << gpu_table[i];
    }
    LOG(INFO) << gpu_dump.str();
  }
#endif

  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_gpu, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
      tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
      tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
      tensor::Tensor block_table_tensor = get_buffer(ModelBufferType::kBlockTable);

      // block_table 缓冲区恒定为 [max_batch_size_, max_blocks_per_req_]
      block_table_tensor.reshape({max_batch_size_, max_blocks_per_req_});
      CHECK_LE(batch_size, max_batch_size_) << "Batch size exceeds preallocated block table rows";
      kernel::batched_write_paged_kv_cu(
          key.ptr<float>(), val.ptr<float>(), pos_gpu.ptr<int32_t>(), block_table_tensor.ptr<int32_t>(),
          const_cast<float*>(key_cache.ptr<float>()), const_cast<float*>(val_cache.ptr<float>()),
          batch_size, config_->kv_dim_, block_size_, max_blocks_per_req_, layer_idx,
          block_manager_.total_blocks(), cuda_config_->stream);
  }
}

/**
 * @brief 执行多头注意力机制的前向计算
 * 
 * @param layer_idx 当前处理的层索引
 * @param pos_tensor 位置张量，包含每个序列的当前位置信息
 */
void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  // 获取批处理大小
  int32_t batch_size = pos_tensor.size(); 
  // 获取 CUDA 设备分配器实例
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  // 获取键缓存和值缓存张量
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  // 准备多头注意力输出张量
  tensor::Tensor mha_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kOutputMHA).ptr<void>()));
  mha_output.reshape({batch_size, config_->dim_}); 
  mha_output.set_device_type(device_type_);

  // 获取注意力分数存储张量
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  
  // 准备查询张量
  tensor::Tensor query(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kQuery).ptr<void>()));
  query.reshape({batch_size, config_->dim_});
  query.set_device_type(device_type_);

  // 获取块表张量（固定 shape: max_batch_size_ × max_blocks_per_req_）
  tensor::Tensor block_table = get_buffer(ModelBufferType::kBlockTable);
  block_table.reshape({max_batch_size_, max_blocks_per_req_});
  CHECK_LE(batch_size, max_batch_size_) << "Batch size exceeds block table capacity";
  
  // 准备 GPU 上的位置张量
  tensor::Tensor pos_gpu(base::DataType::kDataTypeInt32, batch_size, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kPosGPU).ptr<void>()));
  pos_gpu.reshape({batch_size});
  pos_gpu.set_device_type(device_type_);

  ensure_batch_blocks(pos_tensor);

  // 获取多头注意力层并设置参数
  const auto& mha_layer = llama_layers_->mha_layer_;
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_block_size(block_size_);
  
  // 设置多头注意力层的输入和输出
  mha_layer->set_input(0, query);          // 查询张量
  mha_layer->set_input(1, score_storage);  // 注意力分数存储
  mha_layer->set_input(2, key_cache);      // 键缓存
  mha_layer->set_input(3, val_cache);      // 值缓存
  mha_layer->set_input(4, block_table);    // 块表（包含全部请求的映射表）
  mha_layer->set_input(5, pos_gpu);        // 位置张量
  mha_layer->set_output(0, mha_output);    // 多头注意力输出
  
  // 执行多头注意力前向计算
  STATUS_CHECK(mha_layer->forward());

  // 准备注意力输出张量
  tensor::Tensor attn_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kAttnOutput).ptr<void>()));
  attn_output.reshape({batch_size, config_->dim_}); 
  attn_output.set_device_type(device_type_);

  // 获取输出权重层并执行前向计算
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  int32_t batch_size = input.size() / config_->dim_;
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor attn_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kAttnOutput).ptr<void>()));
  attn_output.reshape({batch_size, config_->dim_});
  attn_output.set_device_type(device_type_);
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, attn_output, input));

  tensor::Tensor ffn_norm_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kFFNRMSNorm).ptr<void>()));
  ffn_norm_output.reshape({batch_size, config_->dim_});
  ffn_norm_output.set_device_type(device_type_);

  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  // --- 核心修复 2：FFN 的归一化切片 ---
  for (int i = 0; i < batch_size; ++i) {
    tensor::Tensor in_view(input.data_type(), config_->dim_, false, alloc_cu, const_cast<float*>(input.ptr<float>()) + i * config_->dim_);
    tensor::Tensor out_view(ffn_norm_output.data_type(), config_->dim_, false, alloc_cu, ffn_norm_output.ptr<float>() + i * config_->dim_);
    in_view.set_device_type(device_type_);
    out_view.set_device_type(device_type_);
    STATUS_CHECK(ffn_rmsnorm->forward(in_view, out_view));
  }
  // STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, batch_size * config_->hidden_dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kW1Output).ptr<void>()));
  w1_output.reshape({batch_size, config_->hidden_dim_});
  w1_output.set_device_type(device_type_);

  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, batch_size * config_->hidden_dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kW3Output).ptr<void>()));
  w3_output.reshape({batch_size, config_->hidden_dim_});
  w3_output.set_device_type(device_type_);

  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

  // --- 核心修复 3：强行切片 SwiGLU，避免其退化为单并发导致激活错乱 ---
  // for (int i = 0; i < batch_size; ++i) {
  //   tensor::Tensor w1_view(w1_output.data_type(), config_->hidden_dim_, false, alloc_cu, w1_output.ptr<float>() + i * config_->hidden_dim_);
  //   tensor::Tensor w3_view(w3_output.data_type(), config_->hidden_dim_, false, alloc_cu, w3_output.ptr<float>() + i * config_->hidden_dim_);
  //   w1_view.set_device_type(device_type_);
  //   w3_view.set_device_type(device_type_);
  //   STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_view, w3_view, w1_view));
  // }

  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));

  tensor::Tensor w2_output(base::DataType::kDataTypeFp32, batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kW2Output).ptr<void>()));
  w2_output.reshape({batch_size, config_->dim_});
  w2_output.set_device_type(device_type_);

  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  int32_t batch_size = input.size() / config_->dim_;
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_2d(input.data_type(), batch_size * config_->dim_, false, alloc_cu, const_cast<void*>(input.ptr<void>()));
  input_2d.reshape({batch_size, config_->dim_});
  input_2d.set_device_type(input.device_type());

  //STATUS_CHECK(norm->forward(input_2d, input_2d));
  // --- 核心修复 4：顶层输出分类前的最后一次归一化隔离 ---
  for (int i = 0; i < batch_size; ++i) {
    tensor::Tensor in_view(input.data_type(), config_->dim_, false, alloc_cu, const_cast<float*>(input.ptr<float>()) + i * config_->dim_);
    in_view.set_device_type(device_type_);
    STATUS_CHECK(norm->forward(in_view, in_view)); 
  }

  input_2d.reshape({batch_size, config_->dim_}); // 防止 norm 再次压扁它

  int32_t vocab_size = std::abs(config_->vocab_size_);
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, batch_size * vocab_size, false, alloc_cu, const_cast<void*>(get_buffer(ModelBufferType::kForwardOutput).ptr<void>()));
  forward_output.reshape({batch_size, vocab_size});  
  forward_output.set_device_type(device_type_);

  STATUS_CHECK(llama_layers_->cls_layer_->forward(input_2d, forward_output));
}

base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) return status;
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    int vocab_size = std::abs(config_->vocab_size_);
    bool has_history = !sampling_history_.empty();
    bool apply_penalty = has_history && sampling_config_.repetition_penalty != 1.0f;
    std::vector<float> host_logits;
    auto ensure_host_logits = [&](std::vector<float>& storage) -> float* {
      if (storage.empty()) {
        storage.resize(vocab_size);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
          cudaError_t copy_status = cudaMemcpy(storage.data(), forward_logits,
                                               static_cast<size_t>(vocab_size) * sizeof(float),
                                               cudaMemcpyDeviceToHost);
          CHECK_EQ(copy_status, cudaSuccess) << "Failed to copy logits to host";
        } else {
          std::copy(forward_logits, forward_logits + vocab_size, storage.begin());
        }
      }
      return storage.data();
    };

    auto greedy_from_logits = [&](const float* logits_ptr) -> int32_t {
      return static_cast<int32_t>(std::distance(
          logits_ptr, std::max_element(logits_ptr, logits_ptr + vocab_size)));
    };

    if (apply_penalty) {
      float* penalty_logits = ensure_host_logits(host_logits);
      float penalty = sampling_config_.repetition_penalty;
      int effective_window = sampling_config_.repetition_window <= 0
                                 ? static_cast<int>(sampling_history_.size())
                                 : sampling_config_.repetition_window;
      int start_index = std::max<int>(0, static_cast<int>(sampling_history_.size()) - effective_window);
      for (int idx = start_index; idx < static_cast<int>(sampling_history_.size()); ++idx) {
        int32_t token_id = sampling_history_[idx];
        if (token_id < 0 || token_id >= vocab_size) continue;
        float& logit = penalty_logits[token_id];
        logit = (logit > 0.f) ? (logit / penalty) : (logit * penalty);
      }
    }

    if (sampling_config_.do_sample) {
      float* logits_data = ensure_host_logits(host_logits);
      float temperature = std::max(0.01f, sampling_config_.temperature);
      if (temperature != 1.0f) {
        for (int i = 0; i < vocab_size; ++i) {
          logits_data[i] /= temperature;
        }
      }

      std::vector<int> sorted_indices(vocab_size);
      std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
      std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int lhs, int rhs) {
        return logits_data[lhs] > logits_data[rhs];
      });

      int top_k = sampling_config_.top_k;
      if (top_k > 0 && top_k < static_cast<int>(sorted_indices.size())) {
        sorted_indices.resize(top_k);
      }

      auto compute_probs = [&](const std::vector<int>& indices) {
        std::vector<float> probs;
        if (indices.empty()) return probs;
        probs.reserve(indices.size());
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int idx : indices) {
          max_logit = std::max(max_logit, logits_data[idx]);
        }
        float sum = 0.f;
        for (int idx : indices) {
          float val = std::exp(logits_data[idx] - max_logit);
          probs.push_back(val);
          sum += val;
        }
        if (sum <= 0.f) {
          float uniform = 1.f / static_cast<float>(std::max<size_t>(1, probs.size()));
          for (float& p : probs) p = uniform;
        } else {
          for (float& p : probs) p /= sum;
        }
        return probs;
      };

      std::vector<float> probs = compute_probs(sorted_indices);
      float nucleus = std::clamp(sampling_config_.top_p, 0.0f, 1.0f);
      if (!sorted_indices.empty() && nucleus > 0.0f && nucleus < 0.999f) {
        std::vector<int> filtered_idx;
        std::vector<float> filtered_probs;
        filtered_idx.reserve(sorted_indices.size());
        filtered_probs.reserve(sorted_indices.size());
        float cumulative = 0.f;
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
          filtered_idx.push_back(sorted_indices[i]);
          filtered_probs.push_back(probs[i]);
          cumulative += probs[i];
          if (cumulative >= nucleus) {
            break;
          }
        }
        float renorm = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.f);
        if (renorm > 0.f) {
          for (float& p : filtered_probs) {
            p /= renorm;
          }
        }
        if (!filtered_idx.empty()) {
          sorted_indices.swap(filtered_idx);
          probs.swap(filtered_probs);
        }
      }

      if (sorted_indices.empty()) {
        const float* logits_cpu = host_logits.empty() ? ensure_host_logits(host_logits) : host_logits.data();
        next = greedy_from_logits(logits_cpu);
      } else {
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        size_t sampled = dist(sampling_rng_);
        sampled = std::min(sampled, sorted_indices.size() - 1);
        next = sorted_indices[sampled];
      }
    } else if (device_type_ == base::DeviceType::kDeviceCUDA && !apply_penalty) {
      next = static_cast<int32_t>(sampler_->sample(
          forward_logits, vocab_size, cuda_config_ ? cuda_config_->stream : nullptr));
    } else {
      const float* logits_cpu = host_logits.empty() ? ensure_host_logits(host_logits) : host_logits.data();
      next = greedy_from_logits(logits_cpu);
    }

#ifdef DEBUG_SINGLE_REQ
    const float* inspect_logits = nullptr;
    if (!host_logits.empty()) {
      inspect_logits = host_logits.data();
    } else if (device_type_ == base::DeviceType::kDeviceCUDA) {
      inspect_logits = ensure_host_logits(host_logits);
    } else {
      inspect_logits = forward_logits;
    }
    std::array<int32_t, 5> top_indices{};
    std::array<float, 5> top_values{};
    top_indices.fill(-1);
    top_values.fill(std::numeric_limits<float>::lowest());
    for (int i = 0; i < vocab_size; ++i) {
      float val = inspect_logits[i];
      for (int k = 0; k < 5; ++k) {
        if (val > top_values[k]) {
          for (int shift = 4; shift > k; --shift) {
            top_values[shift] = top_values[shift - 1];
            top_indices[shift] = top_indices[shift - 1];
          }
          top_values[k] = val;
          top_indices[k] = i;
          break;
        }
      }
    }
    std::ostringstream logits_dump;
    logits_dump << "[single-request][logits] top5:";
    for (int k = 0; k < 5; ++k) {
      logits_dump << " (#" << k << ": id=" << top_indices[k]
                  << " logit=" << top_values[k] << ")";
    }
    logits_dump << " chosen=" << next;
    LOG(INFO) << logits_dump.str();
#endif
  }
  return next;
}

}  // namespace model