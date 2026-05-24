#include "op/mha.h"
#include "kernels/cpu/mha_kernel.h"
#include "kernels/kernels_interface.h"
#include "kernels/cuda/mha_kernel.cuh" // --- 引入我们新写的 Batched MHA 核函数 ---
namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                       int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size),
      total_blocks_pool_(0) {
  reset_input_size(6);  // 输入：Q、Score、K cache、V cache、block_table、pos_tensor
  reset_output_size(1);
}

base::Status MultiHeadAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  const tensor::Tensor& mha_out = this->get_output(0);
  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& score_tensor = this->get_input(1);
  const tensor::Tensor& key_cache_tensor = this->get_input(2);
  const tensor::Tensor& value_cache_tensor = this->get_input(3);
  const tensor::Tensor& block_table_tensor = this->get_input(4);
  const tensor::Tensor& pos_tensor = this->get_input(5); // 提取新增加的 GPU Pos 数组

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  int32_t batch_size = pos_tensor.size();

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    int32_t max_blocks_per_req = block_table_tensor.get_dim(1);

    CHECK_GT(total_blocks_pool_, 0)
        << "total_blocks_pool_ must be configured via set_total_blocks()";

    // 批量分页注意力：Grid = (head_num, batch_size)
    kernel::batched_mha_kernel_cu(
        batch_size, pos_tensor, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
        head_size_, mha_out, query_tensor, score_tensor, key_cache_tensor,
        value_cache_tensor, block_table_tensor, block_size_,
        max_blocks_per_req, total_blocks_pool_, device_type_,
        cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_mha_kernel(device_type_)(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,
                                       head_size_, mha_out, query_tensor, score_tensor,
                                       key_cache_tensor, value_cache_tensor, block_table_tensor, block_size_,
                                       device_type_,
                                       cuda_config_ ? cuda_config_.get() : nullptr);
  }
  return base::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) { this->pos_ = pos; }

void MultiHeadAttention::set_layer_idx(int32_t layer_idx) { this->layer_index_ = layer_idx; }

void MultiHeadAttention::set_block_size(int32_t block_size) { this->block_size_ = block_size; }

void MultiHeadAttention::set_total_blocks(int32_t total_blocks) { this->total_blocks_pool_ = total_blocks; }

base::Status MultiHeadAttention::check() const {
  base::Status status;
  const int32_t input_tensor_num = 6;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    // mha score tensor
    // 豁免 block_table (i=4) 和 pos_tensor (i=5) 的类型安检
    if (i == 4 || i == 5) {
      status = check_tensor(get_input(i), device_type_, base::DataType::kDataTypeInt32);
    }else {
      status = check_tensor(get_input(i), device_type_, data_type_);
    }
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the matmul layer.";
      return status;
    }
  }
  return check_tensor(get_output(0), device_type_, data_type_);
}

}  // namespace op