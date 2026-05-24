#include <base/block_manager.h>
#include <deque>
#include <unordered_map>
#include <vector>

enum class ReqStatus { kWaiting, kPrefilling, kDecoding, kFinished };

struct ScheduleRequest {
    int32_t req_id;
    std::vector<int32_t> prompt_tokens;  // 完整 prompt
    int32_t prompt_len;
    int32_t pos = 0;                     // 当前处理到的位置
    ReqStatus status = ReqStatus::kWaiting;
    std::vector<int32_t> block_ids;      // 已分配的物理块 ID 列表
    std::vector<int32_t> generated_ids;  // 生成的 token
    int32_t max_gen_len = 256;
    double arrival_time;                 // 到达时间戳
};

class Scheduler {
public:
    // 提交新请求到等待队列
    int32_t submit(const std::vector<int32_t>& prompt_tokens, int32_t max_gen_len);

    // 核心：每步调用，返回本步该执行的请求列表
    // 同时处理: 新请求准入(prefill) / 进行中请求续推(decode) / 完成请求回收
    struct StepBatch {
        std::vector<int32_t> active_indices;   // 本轮活跃的 req_id 列表
        std::vector<bool> is_prefill;           // 每项是 prefill 还是 decode
        int32_t total_batch_size;
    };
    StepBatch schedule_step();

    // 标记请求完成，释放其占用的 blocks
    void finish_request(int32_t req_id, base::BlockManager& block_manager);

    // [P/D-分离] 访问器 — 主循环通过这些方法读写请求状态
    ScheduleRequest& get_request(int32_t req_id);
    bool waiting_queue_empty() const;
    int32_t waiting_queue_size() const;
    int32_t active_count() const;

    // [P/D-分离] 为新准入的 prefill 请求分配物理块
    // 返回 false 表示块池不足，该请求应退回等待队列
    bool allocate_blocks(int32_t req_id, base::BlockManager& block_manager,
                         int32_t block_size, int32_t max_gen_steps);

    // 调度器参数（公开，允许外部按需调整）
    int32_t max_active_requests_ = 64;
    int32_t max_prefill_tokens_per_step_ = 2048;  // 单步最多处理多少 prefill token

private:
    std::deque<ScheduleRequest> waiting_queue_;   // 等待队列
    std::unordered_map<int32_t, ScheduleRequest> active_requests_;  // 活跃请求
    int32_t next_req_id_ = 0;
};