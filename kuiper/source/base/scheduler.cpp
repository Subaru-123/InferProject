#include <base/scheduler.h>
#include <glog/logging.h>

// ============================================================
// [P/D-分离] submit — 新请求进入等待队列
// ============================================================
int32_t Scheduler::submit(const std::vector<int32_t>& prompt_tokens,
                          int32_t max_gen_len) {
  ScheduleRequest req;
  req.req_id = next_req_id_++;
  req.prompt_tokens = prompt_tokens;
  req.prompt_len = static_cast<int32_t>(prompt_tokens.size());
  req.max_gen_len = max_gen_len;
  req.pos = 0;
  req.status = ReqStatus::kWaiting;
  waiting_queue_.push_back(req);

  LOG(INFO) << "[Scheduler] submitted req_id=" << req.req_id
            << " prompt_len=" << req.prompt_len
            << " max_gen_len=" << max_gen_len;
  return req.req_id;
}

// ============================================================
// [P/D-分离] schedule_step — 每步调度，返回本轮活跃请求
//
// 策略:
//   1. 统计当前 decode 请求数
//   2. 计算空余 slot（max_active_requests_ - 当前活跃数）
//   3. 从 waiting_queue_ 中取出请求，进入 prefill
//   4. 限制单步 prefill token 总数 ≤ max_prefill_tokens_per_step_
//   5. 短 prompt 优先（减少后续 bubble）
// ============================================================
Scheduler::StepBatch Scheduler::schedule_step() {
  StepBatch batch;

  // ── 阶段 1: 收集所有活跃请求 ──
  batch.active_indices.clear();
  batch.is_prefill.clear();
  int32_t cur_prefill_tokens = 0;

  // 先从 active_requests_ 中取正在进行的请求
  for (auto& kv : active_requests_) {
    int32_t rid = kv.first;
    auto& req = kv.second;
    batch.active_indices.push_back(rid);

    if (req.status == ReqStatus::kPrefilling) {
      batch.is_prefill.push_back(true);
      cur_prefill_tokens += (req.prompt_len - req.pos);  // 剩余 prefill token 数
    } else {
      batch.is_prefill.push_back(false);
    }
  }

  // ── 阶段 2: 从等待队列准入新请求 (prefill) ──
  int32_t free_slots = max_active_requests_
                       - static_cast<int32_t>(active_requests_.size());
  int32_t admitted = 0;

  while (free_slots > 0 && !waiting_queue_.empty()) {
    auto& next_req = waiting_queue_.front();

    // 当前步 prefill token 超限 → 下步再准入
    if (cur_prefill_tokens + next_req.prompt_len
        > max_prefill_tokens_per_step_) {
      break;
    }

    // 准入：从等待队列移到活跃集合
    next_req.status = ReqStatus::kPrefilling;
    next_req.pos = 0;
    int32_t rid = next_req.req_id;
    active_requests_[rid] = next_req;
    waiting_queue_.pop_front();

    batch.active_indices.push_back(rid);
    batch.is_prefill.push_back(true);
    cur_prefill_tokens += next_req.prompt_len;
    free_slots--;
    admitted++;

    LOG(INFO) << "[Scheduler] admitted req_id=" << rid
              << " for prefill, prompt_len=" << next_req.prompt_len;
  }

  batch.total_batch_size = static_cast<int32_t>(batch.active_indices.size());
  return batch;
}

// ============================================================
// [P/D-分离] finish_request — 请求完成，移除（block 回收由外部调用）
// ============================================================
void Scheduler::finish_request(int32_t req_id,
                               base::BlockManager& block_manager) {
  auto it = active_requests_.find(req_id);
  if (it == active_requests_.end()) {
    LOG(WARNING) << "[Scheduler] finish_request: req_id=" << req_id
                 << " not found in active set";
    return;
  }

  auto& req = it->second;
  req.status = ReqStatus::kFinished;

  // 回收该请求分配的所有物理块
  if (!req.block_ids.empty()) {
    block_manager.free_blocks(req.block_ids);
    LOG(INFO) << "[Scheduler] freed " << req.block_ids.size()
              << " blocks for req_id=" << req_id;
    req.block_ids.clear();
  }

  active_requests_.erase(it);
  LOG(INFO) << "[Scheduler] finished req_id=" << req_id
            << " (active=" << active_requests_.size()
            << ", waiting=" << waiting_queue_.size() << ")";
}
