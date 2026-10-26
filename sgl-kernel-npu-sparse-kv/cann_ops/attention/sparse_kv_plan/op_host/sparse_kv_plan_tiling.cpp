/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sparse_kv_plan_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sparse_kv_plan_config.h"

namespace optiling {
namespace {
constexpr uint64_t MIN_HASH_CAPACITY = 32U;
constexpr uint32_t PLAN_BLOCKS = 64U;

uint64_t HashCapacity(int64_t topk) {
  uint64_t target = static_cast<uint64_t>(topk) * 2U;
  uint64_t capacity = MIN_HASH_CAPACITY;
  while (capacity < target) {
    capacity <<= 1U;
  }
  return capacity;
}

ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  OP_LOGE(context->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}
}  // namespace

static ge::graphStatus SparseKvPlanTiling(gert::TilingContext* context) {
  const gert::StorageShape* reqShape = context->GetInputShape(0);
  const gert::StorageShape* topkShape = context->GetInputShape(1);
  const gert::StorageShape* blockTableShape = context->GetInputShape(5);
  const auto* attrs = context->GetAttrs();
  if (reqShape == nullptr || topkShape == nullptr || blockTableShape == nullptr || attrs == nullptr) {
    return Fail(context, "SparseKvPlan requires shapes and attributes");
  }
  const gert::Shape& reqOrigin = reqShape->GetOriginShape();
  const gert::Shape& topkOrigin = topkShape->GetOriginShape();
  const gert::Shape& blockTableOrigin = blockTableShape->GetOriginShape();
  if (reqOrigin.GetDimNum() != 1 || topkOrigin.GetDimNum() != 2 || blockTableOrigin.GetDimNum() != 2) {
    return Fail(context, "SparseKvPlan expects req_ids rank 1 and topk/block_table rank 2");
  }

  const int64_t maxRows = topkOrigin.GetDim(0);
  const int64_t maxRequests = reqOrigin.GetDim(0);
  const int64_t maxNumBlocks = blockTableOrigin.GetDim(1);
  const int64_t topk = *attrs->GetInt(0);
  const int64_t capacity = *attrs->GetInt(1);
  const int64_t maxToken = *attrs->GetInt(2);
  const int64_t blockSize = *attrs->GetInt(3);
  const int64_t hostNumBlocks = *attrs->GetInt(4);
  if (maxRows <= 0 || maxRequests <= 0 || topk <= 0 || capacity <= 0 || maxToken <= 0 || maxNumBlocks <= 0 ||
      blockSize <= 0 || hostNumBlocks <= 0) {
    return Fail(context, "SparseKvPlan received a non-positive shape or attribute");
  }
  if (blockTableOrigin.GetDim(0) != maxRequests || topkOrigin.GetDim(1) != topk) {
    return Fail(context, "SparseKvPlan input shapes do not match request/topk attributes");
  }
  if (topk > std::numeric_limits<int32_t>::max() / 2 || capacity > std::numeric_limits<int32_t>::max() ||
      maxToken > std::numeric_limits<int32_t>::max() || blockSize > std::numeric_limits<int32_t>::max()) {
    return Fail(context, "SparseKvPlan shape exceeds int32 kernel limits");
  }

  const uint64_t hashCapacity = HashCapacity(topk);
  const uint64_t rowElements = 3U * hashCapacity + 2U * static_cast<uint64_t>(capacity) + static_cast<uint64_t>(topk);
  const uint64_t rowBytes = rowElements * sizeof(int32_t);
  const uint64_t alignedRowBytes = (rowBytes + 31U) & ~static_cast<uint64_t>(31U);
  const uint64_t localBytes = sparse_kv_plan_config::PLAN_SCAN_BYTES +
                              (alignedRowBytes <= sparse_kv_plan_config::PLAN_ROW_UB_LIMIT_BYTES ? alignedRowBytes : 0U);
  if (localBytes > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "SparseKvPlan local memory size overflow");
  }
  const uint32_t localMemoryBytes = static_cast<uint32_t>(localBytes);
  if (context->SetDynUBufSize(localMemoryBytes) != ge::GRAPH_SUCCESS) {
    return Fail(context, "SparseKvPlan failed to set dynamic UB size");
  }

  SparseKvPlanTilingData tiling;
  tiling.set_hashCapacity(hashCapacity);
  tiling.set_workspaceRowElements(rowElements);
  tiling.set_maxRows(maxRows);
  tiling.set_topk(topk);
  tiling.set_capacity(capacity);
  tiling.set_maxToken(maxToken);
  tiling.set_maxRequests(maxRequests);
  tiling.set_maxNumBlocks(maxNumBlocks);
  tiling.set_hostNumBlocks(hostNumBlocks);
  tiling.set_blockSize(static_cast<int32_t>(blockSize));
  tiling.set_localMemoryBytes(localMemoryBytes);

  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return Fail(context, "SparseKvPlan cannot query platform information");
  }
  const platform_ascendc::PlatformAscendC platform(platformInfo);
  const uint32_t blockDim = std::min<uint32_t>(PLAN_BLOCKS, platform.GetCoreNumAiv());
  if (blockDim == 0) {
    return Fail(context, "SparseKvPlan found zero AIV cores");
  }
  context->SetBlockDim(blockDim);
  size_t* workspaceSize = context->GetWorkspaceSizes(1);
  workspaceSize[0] = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSparseKvPlan(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SparseKvPlan).Tiling(SparseKvPlanTiling).TilingParse<SparseKvPlanCompileInfo>(ParseSparseKvPlan);
}  // namespace optiling
