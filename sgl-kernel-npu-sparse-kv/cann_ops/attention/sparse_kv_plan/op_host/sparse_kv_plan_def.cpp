/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
namespace {
void AddInt32Input(OpDef& op, const char* name) {
  op.Input(name)
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
}

void AddInt64Input(OpDef& op, const char* name) {
  op.Input(name)
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
}
}  // namespace

class SparseKvPlan : public OpDef {
 public:
  explicit SparseKvPlan(const char* name) : OpDef(name) {
    AddInt64Input(*this, "reqIds");
    AddInt32Input(*this, "topkIndices");
    AddInt32Input(*this, "stablePrefixLens");
    AddInt32Input(*this, "visibleSeqLens");
    AddInt32Input(*this, "tokenToReq");
    AddInt32Input(*this, "blockTable");
    AddInt32Input(*this, "activeRows");
    AddInt64Input(*this, "lastReqIds");
    AddInt32Input(*this, "slotToToken");
    AddInt32Input(*this, "lruSlots");
    AddInt32Input(*this, "currentSlots");
    AddInt32Input(*this, "missCount");
    AddInt32Input(*this, "missTokens");
    AddInt32Input(*this, "missSlots");
    AddInt32Input(*this, "compactWorkspace");

    this->Attr("topk").AttrType(REQUIRED).Int();
    this->Attr("capacity").AttrType(REQUIRED).Int();
    this->Attr("maxToken").AttrType(REQUIRED).Int();
    this->Attr("blockSize").AttrType(REQUIRED).Int();
    this->Attr("hostNumBlocks").AttrType(REQUIRED).Int();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("opFile.value", "sparse_kv_plan_apt");
    this->AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(SparseKvPlan);
}  // namespace ops
