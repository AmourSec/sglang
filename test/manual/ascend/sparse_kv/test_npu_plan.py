from __future__ import annotations

import os

import pytest
import torch

from .npu_harness import (
    NpuPlanState,
    assert_state_matches,
    call_plan,
    load_ops,
    workspace_elements,
)
from .reference import PlanState, plan_reference

torch_npu = pytest.importorskip("torch_npu")


BLOCK_SIZE = 128
TOPK = 2048
MAX_TOKEN = 143_360
SEQUENCE_LENGTHS = (1, 127, 128, 129, 2047, 2048, 2049, 4096, 32_768, 140_000, 143_360)


def _require_npu() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def _full_mode() -> bool:
    return os.getenv("SGL_SPARSE_KV_TEST_MODE") == "full"


def _block_table(requests: int) -> list[list[int]]:
    blocks = (MAX_TOKEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    return [
        list(range((request + 1) * blocks - 1, request * blocks - 1, -1))
        for request in range(requests)
    ]


def _run_and_compare(
    actual: NpuPlanState,
    expected: PlanState,
    *,
    req_ids: list[int],
    topk: list[list[int]],
    stable: list[int],
    visible: list[int],
    token_to_req: list[int],
    block_table: list[list[int]],
    capacity: int,
) -> None:
    rows = len(topk)
    host_num_blocks = len(req_ids) * len(block_table[0])
    tensors = {
        "req_ids": torch.tensor(req_ids, dtype=torch.int64, device="npu"),
        "topk_indices": torch.tensor(topk, dtype=torch.int32, device="npu"),
        "stable_prefix_lens": torch.tensor(stable, dtype=torch.int32, device="npu"),
        "visible_seq_lens": torch.tensor(visible, dtype=torch.int32, device="npu"),
        "token_to_req": torch.tensor(token_to_req, dtype=torch.int32, device="npu"),
        "block_table": torch.tensor(block_table, dtype=torch.int32, device="npu"),
        "active_rows": torch.tensor([rows], dtype=torch.int32, device="npu"),
    }
    call_plan(
        load_ops().plan,
        actual,
        **tensors,
        topk=TOPK,
        capacity=capacity,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=host_num_blocks,
    )
    torch_npu.npu.synchronize()
    plan_reference(
        expected,
        req_ids=req_ids,
        topk_indices=topk,
        stable_prefix_lens=stable,
        visible_seq_lens=visible,
        token_to_req=token_to_req,
        block_table=block_table,
        active_rows=rows,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=host_num_blocks,
    )
    assert_state_matches(actual, expected, len(expected.last_req_ids))


@pytest.mark.parametrize("capacity", [2048, 4096, 8192])
def test_plan_matches_oracle_across_ub_and_gm_workspace(capacity: int) -> None:
    _require_npu()
    actual = NpuPlanState(2, TOPK, capacity)
    expected = PlanState.empty(2, TOPK, capacity)
    table = _block_table(2)
    topk = [list(range(TOPK)), list(range(TOPK - 1, -1, -1))]

    _run_and_compare(
        actual,
        expected,
        req_ids=[101, 202],
        topk=topk,
        stable=[MAX_TOKEN, MAX_TOKEN],
        visible=[MAX_TOKEN, MAX_TOKEN],
        token_to_req=[0, 1],
        block_table=table,
        capacity=capacity,
    )

    mixed = [
        list(range(TOPK // 2, TOPK + TOPK // 2)),
        list(range(TOPK + TOPK // 2 - 1, TOPK // 2 - 1, -1)),
    ]
    _run_and_compare(
        actual,
        expected,
        req_ids=[101, 202],
        topk=mixed,
        stable=[MAX_TOKEN, MAX_TOKEN],
        visible=[MAX_TOKEN, MAX_TOKEN],
        token_to_req=[0, 1],
        block_table=table,
        capacity=capacity,
    )
    _run_and_compare(
        actual,
        expected,
        req_ids=[101, 202],
        topk=[row[::-1] for row in mixed],
        stable=[MAX_TOKEN, MAX_TOKEN],
        visible=[MAX_TOKEN, MAX_TOKEN],
        token_to_req=[0, 1],
        block_table=table,
        capacity=capacity,
    )

    assert workspace_elements(2, TOPK, capacity) == (1 if capacity < 8192 else 61_440)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 31, 32])
@pytest.mark.parametrize("width", [1, 4])
def test_plan_full_glm52_batch_sequence_matrix(batch: int, width: int) -> None:
    if not _full_mode():
        pytest.skip(
            "full matrix is selected by scripts/ascend/run_sparse_kv_950.sh full"
        )
    _require_npu()
    rows = batch * width
    capacity = 4096
    actual = NpuPlanState(rows, TOPK, capacity)
    expected = PlanState.empty(rows, TOPK, capacity)
    table = _block_table(batch)
    token_to_req = [row // width for row in range(rows)]

    for generation, sequence_length in enumerate(SEQUENCE_LENGTHS):
        topk = [
            [token % sequence_length for token in range(row, row + TOPK)]
            for row in range(rows)
        ]
        _run_and_compare(
            actual,
            expected,
            req_ids=[generation * 1000 + request for request in range(batch)],
            topk=topk,
            stable=[sequence_length] * rows,
            visible=[
                max(1, sequence_length - width + row % width + 1) for row in range(rows)
            ],
            token_to_req=token_to_req,
            block_table=table,
            capacity=capacity,
        )


def test_plan_reorder_reuse_duplicate_invalid_future_and_inactive_rows() -> None:
    _require_npu()
    capacity = 4096
    rows = 4
    actual = NpuPlanState(rows, TOPK, capacity)
    expected = PlanState.empty(rows, TOPK, capacity)
    table = _block_table(4)
    table[0][7] = -1
    first = [list(range(row, row + TOPK)) for row in range(rows)]
    first[0][:6] = [3, 3, -1, MAX_TOKEN, 140_001, 7 * BLOCK_SIZE]
    _run_and_compare(
        actual,
        expected,
        req_ids=[11, 22, 33, 44],
        topk=first,
        stable=[MAX_TOKEN] * rows,
        visible=[140_001, 2048, 32_768, 143_360],
        token_to_req=[0, 1, 2, 3],
        block_table=table,
        capacity=capacity,
    )

    reordered = [first[2], first[0], first[1]]
    _run_and_compare(
        actual,
        expected,
        req_ids=[33, 11, 55, 44],
        topk=reordered,
        stable=[1024, 1024, 1024],
        visible=[32_768, 140_001, 2048],
        token_to_req=[0, 1, 2],
        block_table=table,
        capacity=capacity,
    )

    restored = [first[2], first[0], first[1], first[3]]
    _run_and_compare(
        actual,
        expected,
        req_ids=[33, 11, 55, 44],
        topk=restored,
        stable=[1024, 1024, 1024, 1024],
        visible=[32_768, 140_001, 2048, 143_360],
        token_to_req=[0, 1, 2, 3],
        block_table=table,
        capacity=capacity,
    )
