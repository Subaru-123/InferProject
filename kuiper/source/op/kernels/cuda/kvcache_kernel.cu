#include "kvcache_kernel.cuh"
#include <algorithm>
#include <vector>
#include <sstream>
#include <glog/logging.h>

namespace kernel {

__global__ void batched_write_paged_kv_kernel(
    const float* k_src,
    const float* v_src,
    const int32_t* pos_array,
    const int32_t* block_table_2d,
    float* key_cache_pool,
    float* value_cache_pool,
    int32_t batch_size,
    int32_t kv_dim,
    int32_t block_size,
    int32_t max_blocks_per_req,
    int32_t layer_idx,
    int32_t total_blocks) {
  int req_idx = blockIdx.y;
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (req_idx >= batch_size || dim_idx >= kv_dim) {
    return;
  }

  int32_t pos = pos_array[req_idx];
  int32_t logical_block_idx = pos / block_size;
  int32_t token_offset = pos % block_size;

  if (logical_block_idx >= max_blocks_per_req) {
#ifdef DEBUG_SINGLE_REQ
    if (req_idx == 0) {
      printf("[single-request][write_kv] pos=%d logical_block=%d exceeds max=%d\n",
             pos, logical_block_idx, max_blocks_per_req);
    }
#endif
    return;
  }

  const int32_t* block_row = block_table_2d + req_idx * max_blocks_per_req;
  int32_t physical_block_idx = block_row[logical_block_idx];
  if (physical_block_idx < 0 || physical_block_idx >= total_blocks) {
#ifdef DEBUG_SINGLE_REQ
    if (req_idx == 0) {
      printf("[single-request][write_kv] invalid physical block %d at logical %d (max %d)\n",
             physical_block_idx, logical_block_idx, total_blocks);
    }
#endif
    return;
  }

  // 层内偏移 = layer_idx * (总物理块 × block_size × kv_dim)
  int64_t layer_offset = static_cast<int64_t>(layer_idx) * total_blocks * block_size * kv_dim;
  int64_t physical_offset = layer_offset +
                            static_cast<int64_t>(physical_block_idx) * block_size * kv_dim +
                            static_cast<int64_t>(token_offset) * kv_dim + dim_idx;

  int64_t src_offset = static_cast<int64_t>(req_idx) * kv_dim + dim_idx;

  key_cache_pool[physical_offset] = k_src[src_offset];
  value_cache_pool[physical_offset] = v_src[src_offset];
}

void batched_write_paged_kv_cu(
    const float* k_src, 
    const float* v_src, 
    const int32_t* pos_array, 
    const int32_t* block_table_2d, 
    float* key_cache_pool, 
    float* value_cache_pool, 
    int32_t batch_size, 
    int32_t kv_dim, 
    int32_t block_size, 
    int32_t max_blocks_per_req, 
    int32_t layer_idx, 
    int32_t total_blocks,
    cudaStream_t stream) 
{
    // 配置 Kernel 的执行网格
    // 线程块大小：一维 128 个线程（处理 kv_dim 的计算）
    dim3 block(128); 
    // 网格大小：
    // X轴：向上取整覆盖 kv_dim
    // Y轴：batch_size (每个请求分配一个或多个 blockIdx.y)
    dim3 grid((kv_dim + block.x - 1) / block.x, batch_size);

    batched_write_paged_kv_kernel<<<grid, block, 0, stream>>>(
        k_src, 
        v_src, 
        pos_array, 
        block_table_2d, 
        key_cache_pool, 
        value_cache_pool, 
        batch_size, 
        kv_dim, 
        block_size, 
        max_blocks_per_req, 
        layer_idx, 
        total_blocks
    );

#ifdef DEBUG_SINGLE_REQ
    cudaError_t kernel_status = cudaGetLastError();
    CHECK_EQ(kernel_status, cudaSuccess) << "batched_write_paged_kv_kernel launch failed";

    // 拉回 block_table 的首个请求，确认写入后仍合法
    if (batch_size == 1) {
      std::vector<int32_t> host_table(max_blocks_per_req, -1);
      size_t bytes = static_cast<size_t>(max_blocks_per_req) * sizeof(int32_t);
      cudaError_t copy_status = cudaMemcpy(host_table.data(), block_table_2d, bytes, cudaMemcpyDeviceToHost);
      CHECK_EQ(copy_status, cudaSuccess) << "Snapshot block_table in batched_write_paged_kv_cu failed";

      int32_t host_pos = 0;
      cudaError_t pos_status = cudaMemcpy(&host_pos, pos_array, sizeof(int32_t), cudaMemcpyDeviceToHost);
      CHECK_EQ(pos_status, cudaSuccess) << "Snapshot pos_array in batched_write_paged_kv_cu failed";
      int32_t logical_block = host_pos / block_size;
      int32_t token_offset = host_pos % block_size;
      int32_t physical_block = logical_block < max_blocks_per_req ? host_table[logical_block] : -1;

      std::ostringstream oss;
      oss << "[single-request][write_kv wrapper] block table head:";
      int limit = max_blocks_per_req < 8 ? max_blocks_per_req : 8;
      for (int i = 0; i < limit; ++i) {
        oss << " [" << i << "]=" << host_table[i];
      }
      LOG(INFO) << oss.str();

      if (logical_block < max_blocks_per_req && physical_block >= 0) {
        size_t layer_span = static_cast<size_t>(total_blocks) * block_size * kv_dim;
        size_t kv_base = static_cast<size_t>(layer_idx) * layer_span +
                         static_cast<size_t>(physical_block) * block_size * kv_dim +
                         static_cast<size_t>(token_offset) * kv_dim;
        int dump_width = std::min<int32_t>(kv_dim, 8);
        std::vector<float> host_k(dump_width, 0.f);
        std::vector<float> host_v(dump_width, 0.f);
        cudaError_t key_status = cudaMemcpy(host_k.data(), key_cache_pool + kv_base, dump_width * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t val_status = cudaMemcpy(host_v.data(), value_cache_pool + kv_base, dump_width * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK_EQ(key_status, cudaSuccess) << "Snapshot key cache slice failed";
        CHECK_EQ(val_status, cudaSuccess) << "Snapshot value cache slice failed";

        std::ostringstream kv_dump;
        kv_dump << "[single-request][write_kv wrapper] pos=" << host_pos
                << " logical_block=" << logical_block
                << " token_offset=" << token_offset
                << " physical_block=" << physical_block
                << " key_head:";
        for (int i = 0; i < dump_width; ++i) {
          kv_dump << " " << host_k[i];
        }
        kv_dump << " | value_head:";
        for (int i = 0; i < dump_width; ++i) {
          kv_dump << " " << host_v[i];
        }
        LOG(INFO) << kv_dump.str();
      } else {
        LOG(WARNING) << "[single-request][write_kv wrapper] invalid block snapshot, logical_block="
                     << logical_block << " physical_block=" << physical_block;
      }
    }
#endif
}

} // namespace kernel