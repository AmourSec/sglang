/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeSparseKvTransfer(gert::InferShapeContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_INFERSHAPE(SparseKvTransfer).InferShape(InferShapeSparseKvTransfer);
}  // namespace ops
