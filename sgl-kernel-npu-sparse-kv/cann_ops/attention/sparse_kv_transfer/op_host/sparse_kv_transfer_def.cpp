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
}  // namespace

class SparseKvTransfer : public OpDef {
 public:
  explicit SparseKvTransfer(const char* name) : OpDef(name) {
    AddInt32Input(*this, "missCount");
    AddInt32Input(*this, "missTokens");
    AddInt32Input(*this, "missSlots");
    AddInt32Input(*this, "tokenToReq");
    AddInt32Input(*this, "blockTable");
    this->Input("hostCacheBases")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    AddInt32Input(*this, "activeRows");
    this->Input("residentK")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("residentV")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();

    this->Attr("topk").AttrType(REQUIRED).Int();
    this->Attr("capacity").AttrType(REQUIRED).Int();
    this->Attr("blockSize").AttrType(REQUIRED).Int();
    this->Attr("hostNumBlocks").AttrType(REQUIRED).Int();
    this->Attr("tokenSizeBytesK").AttrType(REQUIRED).Int();
    this->Attr("tokenSizeBytesV").AttrType(REQUIRED).Int();

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("opFile.value", "sparse_kv_transfer_apt");
    this->AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(SparseKvTransfer);
}  // namespace ops
