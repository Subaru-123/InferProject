#pragma once
#include "tensor/tensor.h"
#include <vector>
#include <queue>

namespace model {
  struct SequenceState {
    int32_t seq_id;
    int32_t logical_len = 0; //当前序列长度
    std::vector<int32_t> block_table; //逻辑块 -》 物理块映射
  };
  class KVCacheManager {
    public:
      KVCacheManager(int num_blocks, int block_size, int num_players, int kv_dim, base::DeviceType device_type);

      // 为特定序列分配一个新块
      void allocate_block(SequenceState& seq);
      // 释放序列所有块
      void free_blocks(SequenceState& seq);

      // 获取当前序列的 block_table Tensor, 用于传给GPU
      tensor::Tensor get_block_table_tensor(const SequenceState& seq);

      int32_t block_size_;
      int32_t num_blocks_;

      // 物理底座 ：形状变为 [num_layers, num_blocks, block_size, kv_dim]
      tensor::Tensor key_cache_pool_;
      tensor::Tensor value_cache_pool_;
    private:
      std::queue<int32_t> free_block_pool_; // 维护空闲块的ID
  };
}