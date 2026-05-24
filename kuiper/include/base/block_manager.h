#ifndef KUIPER_INCLUIDE_BASE_BLOCK_MANAGER_H_
#define KUIPER_INCLUIDE_BASE_BLOCK_MANAGER_H_

#include <vector>
#include <numeric>
#include <cstdint>
#include <glog/logging.h>

namespace base {
  class BlockManager {
    public:
      BlockManager() = default;

      // 初始化全局物理块池
      void init(int32_t total_physical_blocks) {
        CHECK_GT(total_physical_blocks, 0) << "total_physical_blocks must be > 0";
        total_blocks_ = total_physical_blocks;
        allocated_blocks_ = 0;
        free_blocks_.clear();
        free_blocks_.reserve(total_physical_blocks);
        in_use_.assign(total_physical_blocks, false);

        // 生成逆序物理块索引：[N-1, N-2, ..., 1, 0]
        // pop_back() 时优先分配索引靠前的块
        for (int32_t i = total_physical_blocks - 1; i >= 0; --i) {
          free_blocks_.push_back(i);
        }
      }

      inline int32_t allocate_block() {
        CHECK_GT(total_blocks_, 0) << "BlockManager has not been initialized.";
        CHECK(!free_blocks_.empty()) << "No free blocks available for allocation!";

        int32_t block_id = free_blocks_.back();
        free_blocks_.pop_back();

        CHECK(!in_use_[block_id]) << "Block " << block_id << " double allocated.";
        in_use_[block_id] = true;
        ++allocated_blocks_;
        return block_id;
      }

      // 极简回收：直接压回栈顶
      inline void free_block(int32_t block_id) {
        CHECK(block_id >= 0 && block_id < total_blocks_) << "Invalid block id: " << block_id;
        CHECK(in_use_[block_id]) << "Block " << block_id << " double freed.";

        in_use_[block_id] = false;
        free_blocks_.push_back(block_id);
        --allocated_blocks_;
      }
      
      // 批量回收 (用于请求结束时)
      inline void free_blocks(const std::vector<int32_t>& block_ids) {
        for (int32_t id : block_ids) {
          free_block(id);
        }
      }

      inline int32_t total_blocks() const { return total_blocks_; }
      inline int32_t allocated_blocks() const { return allocated_blocks_; }
      inline int32_t free_block_count() const { return static_cast<int32_t>(free_blocks_.size()); }

    private:
      std::vector<int32_t> free_blocks_;  // 全局物理块池
      std::vector<uint8_t> in_use_;
      int32_t total_blocks_ = 0;
      int32_t allocated_blocks_ = 0;
  };
}  //namespace base

#endif  // KUIPER_INCLUIDE_BASE_BLOCK_MANAGER_H_