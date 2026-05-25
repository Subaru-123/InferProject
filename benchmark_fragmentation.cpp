#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>

// 严格按照数学公式计算原版连续内存的浪费率
void calc_continuous_allocation(const std::vector<int>& seq_lens, int max_seq_len, int kv_dim) {
    int total_requests = seq_lens.size();
    
    // 总分配量 (按 max_seq_len 算死)
    int64_t total_allocated_tokens = total_requests * max_seq_len;
    
    // 实际有效数据量
    int64_t total_used_tokens = 0;
    for (int len : seq_lens) {
        total_used_tokens += len;
    }
    
    int64_t wasted_tokens = total_allocated_tokens - total_used_tokens;
    double waste_ratio = static_cast<double>(wasted_tokens) / total_allocated_tokens * 100.0;
    
    std::cout << "[连续内存分配]" << std::endl;
    std::cout << "总分配 Token 数: " << total_allocated_tokens << std::endl;
    std::cout << "实际使用 Token 数: " << total_used_tokens << std::endl;
    std::cout << "浪费 Token 数: " << wasted_tokens << std::endl;
    std::cout << "显存浪费率: " << std::fixed << std::setprecision(2) << waste_ratio << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// 严格按照数学公式计算 Paged Attention 的浪费率
void calc_paged_allocation(const std::vector<int>& seq_lens, int block_size, int kv_dim) {
    int64_t total_allocated_tokens = 0;
    int64_t total_used_tokens = 0;
    
    for (int len : seq_lens) {
        total_used_tokens += len;
        // 代数推导：计算该序列需要的物理块数量 (向上取整)
        int blocks_needed = (len + block_size - 1) / block_size;
        total_allocated_tokens += blocks_needed * block_size;
    }
    
    int64_t wasted_tokens = total_allocated_tokens - total_used_tokens;
    double waste_ratio = static_cast<double>(wasted_tokens) / total_allocated_tokens * 100.0;
    
    std::cout << "[Paged Attention]" << std::endl;
    std::cout << "Block Size: " << block_size << std::endl;
    std::cout << "总分配 Token 数: " << total_allocated_tokens << std::endl;
    std::cout << "实际使用 Token 数: " << total_used_tokens << std::endl;
    std::cout << "浪费 Token 数: " << wasted_tokens << std::endl;
    std::cout << "显存浪费率: " << std::fixed << std::setprecision(2) << waste_ratio << "%" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
    // 模拟一组真实场景中的请求长度（长短不一的对话）
    std::vector<int> seq_lens = {
        12, 45, 128, 7, 256, 14, 88, 30, 2, 400, 
        15, 60, 200, 9, 150, 22, 77, 34, 5, 310
    };
    
    // 假设系统配置的最大序列长度和 KV 维度
    int max_seq_len = 512; 
    int kv_dim = 128; // KV Cache 隐藏层维度
    int block_size = 16;
    
    std::cout << "========== KV Cache 显存碎片率基准测试 ==========" << std::endl;
    std::cout << "模拟并发请求数: " << seq_lens.size() << std::endl;
    std::cout << "系统预设最大上下文长度 (max_seq_len): " << max_seq_len << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    calc_continuous_allocation(seq_lens, max_seq_len, kv_dim);
    calc_paged_allocation(seq_lens, block_size, kv_dim);
    
    return 0;
}