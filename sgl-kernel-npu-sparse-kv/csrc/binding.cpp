// Copyright (c) 2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string_view>

#include "aclnn_torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace sgl_sparse_kv {
namespace {

constexpr int64_t kMinHashCapacity = 32;
constexpr int64_t kPlanRowUbLimitBytes = 112 * 1024;

void CheckTensor(const at::Tensor& tensor, std::string_view name, at::ScalarType dtype, int64_t rank) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == rank, name, " must have rank ", rank);
}

void CheckVector(const at::Tensor& tensor, std::string_view name, at::ScalarType dtype, int64_t size) {
  CheckTensor(tensor, name, dtype, 1);
  TORCH_CHECK(tensor.size(0) == size, name, " must have shape [", size, "]");
}

void CheckMatrix(const at::Tensor& tensor, std::string_view name, int64_t rows, int64_t columns) {
  CheckTensor(tensor, name, at::kInt, 2);
  TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) == columns, name, " has an invalid shape");
}

void CheckSameDevice(const c10::Device& device, std::initializer_list<at::Tensor> tensors) {
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device, "all tensor inputs must be on the same NPU device");
  }
}

int64_t HashCapacity(int64_t topk) {
  TORCH_CHECK(topk <= std::numeric_limits<int32_t>::max() / 2, "topk exceeds the int32 kernel limit");
  int64_t result = kMinHashCapacity;
  while (result < 2 * topk) {
    result *= 2;
  }
  return result;
}

void CheckPositiveAttributes(int64_t topk, int64_t capacity, int64_t block_size, int64_t host_num_blocks) {
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(capacity > 0, "capacity must be positive");
  TORCH_CHECK(topk <= capacity, "topk must not exceed capacity");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(host_num_blocks > 0, "host_num_blocks must be positive");
  TORCH_CHECK(capacity <= std::numeric_limits<int32_t>::max(), "capacity exceeds the int32 kernel limit");
  TORCH_CHECK(block_size <= std::numeric_limits<int32_t>::max(),
              "block_size exceeds the int32 kernel limit");
}

void ValidatePlan(const at::Tensor& req_ids, const at::Tensor& topk_indices,
                  const at::Tensor& stable_prefix_lens, const at::Tensor& visible_seq_lens,
                  const at::Tensor& token_to_req, const at::Tensor& block_table,
                  const at::Tensor& active_rows, const at::Tensor& last_req_ids,
                  const at::Tensor& slot_to_token, const at::Tensor& lru_slots,
                  const at::Tensor& current_slots, const at::Tensor& miss_count,
                  const at::Tensor& miss_tokens, const at::Tensor& miss_slots,
                  const at::Tensor& compact_workspace, int64_t topk, int64_t capacity,
                  int64_t max_token, int64_t block_size, int64_t host_num_blocks) {
  CheckPositiveAttributes(topk, capacity, block_size, host_num_blocks);
  TORCH_CHECK(max_token > 0 && max_token <= std::numeric_limits<int32_t>::max(), "max_token is out of range");
  CheckTensor(req_ids, "req_ids", at::kLong, 1);
  CheckTensor(topk_indices, "topk_indices", at::kInt, 2);
  const int64_t rows = topk_indices.size(0);
  const int64_t requests = req_ids.size(0);
  TORCH_CHECK(rows > 0 && requests > 0, "rows and requests must be positive");
  TORCH_CHECK(topk_indices.size(1) == topk, "topk_indices does not match topk");
  CheckVector(stable_prefix_lens, "stable_prefix_lens", at::kInt, rows);
  CheckVector(visible_seq_lens, "visible_seq_lens", at::kInt, rows);
  CheckVector(token_to_req, "token_to_req", at::kInt, rows);
  CheckTensor(block_table, "block_table", at::kInt, 2);
  TORCH_CHECK(block_table.size(0) == requests && block_table.size(1) > 0, "block_table has an invalid shape");
  CheckVector(active_rows, "active_rows", at::kInt, 1);
  CheckVector(last_req_ids, "last_req_ids", at::kLong, rows);
  CheckMatrix(slot_to_token, "slot_to_token", rows, capacity);
  CheckMatrix(lru_slots, "lru_slots", rows, capacity);
  CheckMatrix(current_slots, "current_slots", rows, topk);
  CheckVector(miss_count, "miss_count", at::kInt, rows);
  CheckMatrix(miss_tokens, "miss_tokens", rows, topk);
  CheckMatrix(miss_slots, "miss_slots", rows, topk);
  CheckTensor(compact_workspace, "compact_workspace", at::kInt, 1);
  const int64_t row_elements = 3 * HashCapacity(topk) + 2 * capacity + topk;
  TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / row_elements, "workspace size overflows int64");
  const int64_t required_elements =
      row_elements * static_cast<int64_t>(sizeof(int32_t)) <= kPlanRowUbLimitBytes ? 1 : rows * row_elements;
  TORCH_CHECK(compact_workspace.numel() >= required_elements, "compact_workspace is too small");
  CheckSameDevice(req_ids.device(),
                  {topk_indices, stable_prefix_lens, visible_seq_lens, token_to_req, block_table,
                   active_rows, last_req_ids, slot_to_token, lru_slots, current_slots, miss_count,
                   miss_tokens, miss_slots, compact_workspace});
}

void ValidateResident(const at::Tensor& resident, std::string_view name, int64_t rows, int64_t capacity,
                      int64_t token_size_bytes) {
  TORCH_CHECK(token_size_bytes > 0 && token_size_bytes <= std::numeric_limits<int32_t>::max(),
              name, " token size is out of range");
  CheckTensor(resident, name, at::kBFloat16, 4);
  TORCH_CHECK(resident.size(0) == rows && resident.size(1) == capacity && resident.size(2) == 1,
              name, " must have shape [rows, capacity, 1, dim]");
  TORCH_CHECK(token_size_bytes % resident.element_size() == 0 &&
                  resident.size(3) == token_size_bytes / resident.element_size(),
              name, " last dimension does not match token_size_bytes");
  TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / capacity,
              name, " element count overflows int64");
  const int64_t slots = rows * capacity;
  TORCH_CHECK(slots <= std::numeric_limits<int64_t>::max() / token_size_bytes,
              name, " byte size overflows int64");
  TORCH_CHECK(resident.nbytes() == rows * capacity * token_size_bytes, name, " byte size is inconsistent");
}

void CheckHostByteRange(int64_t host_num_blocks, int64_t block_size, int64_t token_size_bytes) {
  const uint64_t blocks = static_cast<uint64_t>(host_num_blocks);
  const uint64_t tokens_per_block = static_cast<uint64_t>(block_size);
  const uint64_t token_bytes = static_cast<uint64_t>(token_size_bytes);
  TORCH_CHECK(tokens_per_block <= std::numeric_limits<uint64_t>::max() / token_bytes,
              "host cache block byte size overflows uint64");
  const uint64_t block_bytes = tokens_per_block * token_bytes;
  TORCH_CHECK(blocks <= std::numeric_limits<uint64_t>::max() / block_bytes,
              "host cache byte range overflows uint64");
}

void ValidateTransfer(const at::Tensor& miss_count, const at::Tensor& miss_tokens,
                      const at::Tensor& miss_slots, const at::Tensor& token_to_req,
                      const at::Tensor& block_table, const at::Tensor& host_cache_bases,
                      const at::Tensor& active_rows, const at::Tensor& resident_k,
                      const at::Tensor& resident_v, int64_t topk, int64_t capacity,
                      int64_t block_size, int64_t host_num_blocks, int64_t token_size_bytes_k,
                      int64_t token_size_bytes_v) {
  CheckPositiveAttributes(topk, capacity, block_size, host_num_blocks);
  CheckTensor(miss_count, "miss_count", at::kInt, 1);
  const int64_t rows = miss_count.size(0);
  TORCH_CHECK(rows > 0, "rows must be positive");
  CheckMatrix(miss_tokens, "miss_tokens", rows, topk);
  CheckMatrix(miss_slots, "miss_slots", rows, topk);
  CheckVector(token_to_req, "token_to_req", at::kInt, rows);
  CheckTensor(block_table, "block_table", at::kInt, 2);
  TORCH_CHECK(block_table.size(0) > 0 && block_table.size(1) > 0, "block_table has an invalid shape");
  CheckVector(host_cache_bases, "host_cache_bases", at::kLong, 2);
  CheckVector(active_rows, "active_rows", at::kInt, 1);
  ValidateResident(resident_k, "resident_k", rows, capacity, token_size_bytes_k);
  ValidateResident(resident_v, "resident_v", rows, capacity, token_size_bytes_v);
  CheckHostByteRange(host_num_blocks, block_size, token_size_bytes_k);
  CheckHostByteRange(host_num_blocks, block_size, token_size_bytes_v);
  CheckSameDevice(miss_count.device(),
                  {miss_tokens, miss_slots, token_to_req, block_table, host_cache_bases,
                   active_rows, resident_k, resident_v});
}

void Plan(const at::Tensor& req_ids, const at::Tensor& topk_indices,
          const at::Tensor& stable_prefix_lens, const at::Tensor& visible_seq_lens,
          const at::Tensor& token_to_req, const at::Tensor& block_table,
          const at::Tensor& active_rows, const at::Tensor& last_req_ids,
          const at::Tensor& slot_to_token, const at::Tensor& lru_slots,
          const at::Tensor& current_slots, const at::Tensor& miss_count,
          const at::Tensor& miss_tokens, const at::Tensor& miss_slots,
          const at::Tensor& compact_workspace, int64_t topk, int64_t capacity,
          int64_t max_token, int64_t block_size, int64_t host_num_blocks) {
  ValidatePlan(req_ids, topk_indices, stable_prefix_lens, visible_seq_lens, token_to_req, block_table,
               active_rows, last_req_ids, slot_to_token, lru_slots, current_slots, miss_count,
               miss_tokens, miss_slots, compact_workspace, topk, capacity, max_token, block_size,
               host_num_blocks);
  const c10::DeviceGuard device_guard(req_ids.device());
  EXEC_NPU_CMD(aclnnSparseKvPlan, req_ids, topk_indices, stable_prefix_lens, visible_seq_lens,
               token_to_req, block_table, active_rows, last_req_ids, slot_to_token, lru_slots,
               current_slots, miss_count, miss_tokens, miss_slots, compact_workspace, topk,
               capacity, max_token, block_size, host_num_blocks);
}

void Transfer(const at::Tensor& miss_count, const at::Tensor& miss_tokens,
              const at::Tensor& miss_slots, const at::Tensor& token_to_req,
              const at::Tensor& block_table, const at::Tensor& host_cache_bases,
              const at::Tensor& active_rows, const at::Tensor& resident_k,
              const at::Tensor& resident_v, int64_t topk, int64_t capacity,
              int64_t block_size, int64_t host_num_blocks, int64_t token_size_bytes_k,
              int64_t token_size_bytes_v) {
  ValidateTransfer(miss_count, miss_tokens, miss_slots, token_to_req, block_table, host_cache_bases,
                   active_rows, resident_k, resident_v, topk, capacity, block_size, host_num_blocks,
                   token_size_bytes_k, token_size_bytes_v);
  const c10::DeviceGuard device_guard(miss_count.device());
  EXEC_NPU_CMD(aclnnSparseKvTransfer, miss_count, miss_tokens, miss_slots, token_to_req, block_table,
               host_cache_bases, active_rows, resident_k, resident_v, topk, capacity, block_size,
               host_num_blocks, token_size_bytes_k, token_size_bytes_v);
}

void PlanTransfer(const at::Tensor& req_ids, const at::Tensor& topk_indices,
                  const at::Tensor& stable_prefix_lens, const at::Tensor& visible_seq_lens,
                  const at::Tensor& token_to_req, const at::Tensor& block_table,
                  const at::Tensor& active_rows, const at::Tensor& last_req_ids,
                  const at::Tensor& slot_to_token, const at::Tensor& lru_slots,
                  const at::Tensor& current_slots, const at::Tensor& miss_count,
                  const at::Tensor& miss_tokens, const at::Tensor& miss_slots,
                  const at::Tensor& compact_workspace, const at::Tensor& host_cache_bases,
                  const at::Tensor& resident_k, const at::Tensor& resident_v, int64_t topk,
                  int64_t capacity, int64_t max_token, int64_t block_size,
                  int64_t host_num_blocks, int64_t token_size_bytes_k, int64_t token_size_bytes_v) {
  ValidatePlan(req_ids, topk_indices, stable_prefix_lens, visible_seq_lens, token_to_req, block_table,
               active_rows, last_req_ids, slot_to_token, lru_slots, current_slots, miss_count,
               miss_tokens, miss_slots, compact_workspace, topk, capacity, max_token, block_size,
               host_num_blocks);
  ValidateTransfer(miss_count, miss_tokens, miss_slots, token_to_req, block_table, host_cache_bases,
                   active_rows, resident_k, resident_v, topk, capacity, block_size, host_num_blocks,
                   token_size_bytes_k, token_size_bytes_v);
  Plan(req_ids, topk_indices, stable_prefix_lens, visible_seq_lens, token_to_req, block_table,
       active_rows, last_req_ids, slot_to_token, lru_slots, current_slots, miss_count,
       miss_tokens, miss_slots, compact_workspace, topk, capacity, max_token, block_size,
       host_num_blocks);
  Transfer(miss_count, miss_tokens, miss_slots, token_to_req, block_table, host_cache_bases,
           active_rows, resident_k, resident_v, topk, capacity, block_size, host_num_blocks,
           token_size_bytes_k, token_size_bytes_v);
}

void MetaPlan(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
              const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t,
              int64_t, int64_t, int64_t) {}

void MetaTransfer(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}

void MetaPlanTransfer(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                      int64_t, int64_t, int64_t) {}

}
}

TORCH_LIBRARY(sgl_sparse_kv, ops) {
  ops.def("plan(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, Tensor visible_seq_lens, "
          "Tensor token_to_req, Tensor block_table, Tensor active_rows, Tensor(a!) last_req_ids, "
          "Tensor(b!) slot_to_token, Tensor(c!) lru_slots, Tensor(d!) current_slots, Tensor(e!) miss_count, "
          "Tensor(f!) miss_tokens, Tensor(g!) miss_slots, Tensor(h!) compact_workspace, int topk, "
          "int capacity, int max_token, int block_size, int host_num_blocks) -> ()");
  ops.def("transfer(Tensor miss_count, Tensor miss_tokens, Tensor miss_slots, Tensor token_to_req, "
          "Tensor block_table, Tensor host_cache_bases, Tensor active_rows, Tensor(a!) resident_k, "
          "Tensor(b!) resident_v, int topk, int capacity, int block_size, int host_num_blocks, "
          "int token_size_bytes_k, int token_size_bytes_v) -> ()");
  ops.def("plan_transfer(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
          "Tensor visible_seq_lens, Tensor token_to_req, Tensor block_table, Tensor active_rows, "
          "Tensor(a!) last_req_ids, Tensor(b!) slot_to_token, Tensor(c!) lru_slots, "
          "Tensor(d!) current_slots, Tensor(e!) miss_count, Tensor(f!) miss_tokens, "
          "Tensor(g!) miss_slots, Tensor(h!) compact_workspace, Tensor host_cache_bases, "
          "Tensor(i!) resident_k, Tensor(j!) resident_v, int topk, int capacity, int max_token, "
          "int block_size, int host_num_blocks, int token_size_bytes_k, int token_size_bytes_v) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_sparse_kv, PrivateUse1, ops) {
  ops.impl("plan", &sgl_sparse_kv::Plan);
  ops.impl("transfer", &sgl_sparse_kv::Transfer);
  ops.impl("plan_transfer", &sgl_sparse_kv::PlanTransfer);
}

TORCH_LIBRARY_IMPL(sgl_sparse_kv, Meta, ops) {
  ops.impl("plan", &sgl_sparse_kv::MetaPlan);
  ops.impl("transfer", &sgl_sparse_kv::MetaTransfer);
  ops.impl("plan_transfer", &sgl_sparse_kv::MetaPlanTransfer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
