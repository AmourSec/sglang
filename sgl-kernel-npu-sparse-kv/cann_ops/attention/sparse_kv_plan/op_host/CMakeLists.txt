# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE sparse_kv_plan_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME SparseKvPlan
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
            -mllvm -cce-aicore-dcci-before-kernel-end=false
)

add_modules_sources_with_soc(OPTYPE sparse_kv_plan ACLNNTYPE aclnn)
