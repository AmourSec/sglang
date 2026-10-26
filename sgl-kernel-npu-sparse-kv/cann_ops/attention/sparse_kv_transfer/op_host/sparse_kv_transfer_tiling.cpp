/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sparse_kv_transfer_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t TRANSFER_BLOCKS = 64U;

ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  OP_LOGE(context->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}
}  // namespace

static ge::graphStatus SparseKvTransferTiling(gert::TilingContext* context) {
  const gert::StorageShape* missShape = context->GetInputShape(0);
  const gert::StorageShape* blockTableShape = context->GetInputShape(4);
  const auto* attrs = context->GetAttrs();
  if (missShape == nullptr || blockTableShape == nullptr || attrs == nullptr) {
    return Fail(context, "SparseKvTransfer requires shapes and attributes");
  }
  const gert::Shape& missOrigin = missShape->GetOriginShape();
  const gert::Shape& blockTableOrigin = blockTableShape->GetOriginShape();
  if (missOrigin.GetDimNum() != 1 || blockTableOrigin.GetDimNum() != 2) {
    return Fail(context, "SparseKvTransfer expects miss_count rank 1 and block_table rank 2");
  }

  const int64_t maxRows = missOrigin.GetDim(0);
  const int64_t maxRequests = blockTableOrigin.GetDim(0);
  const int64_t maxNumBlocks = blockTableOrigin.GetDim(1);
  const int64_t topk = *attrs->GetInt(0);
  const int64_t capacity = *attrs->GetInt(1);
  const int64_t blockSize = *attrs->GetInt(2);
  const int64_t hostNumBlocks = *attrs->GetInt(3);
  const int64_t tokenSizeBytesK = *attrs->GetInt(4);
  const int64_t tokenSizeBytesV = *attrs->GetInt(5);
  if (maxRows <= 0 || maxRequests <= 0 || maxNumBlocks <= 0 || topk <= 0 || capacity <= 0 || blockSize <= 0 ||
      hostNumBlocks <= 0 || tokenSizeBytesK <= 0 || tokenSizeBytesV <= 0) {
    return Fail(context, "SparseKvTransfer received a non-positive shape or attribute");
  }
  if (blockSize > std::numeric_limits<int32_t>::max() ||
      tokenSizeBytesK > std::numeric_limits<int32_t>::max() ||
      tokenSizeBytesV > std::numeric_limits<int32_t>::max()) {
    return Fail(context, "SparseKvTransfer token size exceeds int32 kernel limits");
  }
  const uint64_t blocks = static_cast<uint64_t>(hostNumBlocks);
  const uint64_t tokensPerBlock = static_cast<uint64_t>(blockSize);
  const uint64_t tokenBytesK = static_cast<uint64_t>(tokenSizeBytesK);
  const uint64_t tokenBytesV = static_cast<uint64_t>(tokenSizeBytesV);
  if (tokensPerBlock > std::numeric_limits<uint64_t>::max() / tokenBytesK ||
      tokensPerBlock > std::numeric_limits<uint64_t>::max() / tokenBytesV) {
    return Fail(context, "SparseKvTransfer block byte size overflows uint64");
  }
  if (blocks > std::numeric_limits<uint64_t>::max() / (tokensPerBlock * tokenBytesK) ||
      blocks > std::numeric_limits<uint64_t>::max() / (tokensPerBlock * tokenBytesV)) {
    return Fail(context, "SparseKvTransfer host cache byte range overflows uint64");
  }

  SparseKvTransferTilingData tiling;
  tiling.set_maxRows(maxRows);
  tiling.set_maxRequests(maxRequests);
  tiling.set_maxNumBlocks(maxNumBlocks);
  tiling.set_topk(topk);
  tiling.set_capacity(capacity);
  tiling.set_hostNumBlocks(hostNumBlocks);
  tiling.set_blockSize(static_cast<int32_t>(blockSize));
  tiling.set_tokenSizeBytesK(static_cast<int32_t>(tokenSizeBytesK));
  tiling.set_tokenSizeBytesV(static_cast<int32_t>(tokenSizeBytesV));

  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return Fail(context, "SparseKvTransfer cannot query platform information");
  }
  const platform_ascendc::PlatformAscendC platform(platformInfo);
  const uint32_t blockDim = std::min<uint32_t>(TRANSFER_BLOCKS, platform.GetCoreNumAiv());
  if (blockDim == 0) {
    return Fail(context, "SparseKvTransfer found zero AIV cores");
  }
  context->SetBlockDim(blockDim);
  size_t* workspaceSize = context->GetWorkspaceSizes(1);
  workspaceSize[0] = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSparseKvTransfer(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SparseKvTransfer)
    .Tiling(SparseKvTransferTiling)
    .TilingParse<SparseKvTransferCompileInfo>(ParseSparseKvTransfer);
}  // namespace optiling
