from __future__ import annotations

import pytest
import torch

from .npu_harness import workspace_elements
from .reference import PlanState, plan_reference, transfer_reference


def _identity_table(requests: int, tokens: int, block_size: int) -> list[list[int]]:
    blocks = (tokens + block_size - 1) // block_size
    return [
        list(range(request * blocks, (request + 1) * blocks))
        for request in range(requests)
    ]


def _run(
    state: PlanState,
    topk: list[list[int]],
    *,
    req_ids: list[int] | None = None,
    stable: list[int] | None = None,
    visible: list[int] | None = None,
    token_to_req: list[int] | None = None,
    block_table: list[list[int]] | None = None,
    max_token: int = 64,
    block_size: int = 16,
) -> None:
    rows = len(topk)
    requests = rows if req_ids is None else len(req_ids)
    plan_reference(
        state,
        req_ids=list(range(100, 100 + requests)) if req_ids is None else req_ids,
        topk_indices=topk,
        stable_prefix_lens=[max_token] * rows if stable is None else stable,
        visible_seq_lens=[max_token] * rows if visible is None else visible,
        token_to_req=list(range(rows)) if token_to_req is None else token_to_req,
        block_table=_identity_table(requests, max_token, block_size)
        if block_table is None
        else block_table,
        active_rows=rows,
        max_token=max_token,
        block_size=block_size,
        host_num_blocks=requests * ((max_token + block_size - 1) // block_size),
    )


def test_plan_cold_all_hit_mixed_and_lru_eviction() -> None:
    state = PlanState.empty(rows=1, topk=4, capacity=4)

    _run(state, [[0, 1, 2, 3]])
    assert state.miss_count == [4]
    assert state.current_slots == [[0, 1, 2, 3]]
    assert state.lru_slots == [[0, 1, 2, 3]]

    _run(state, [[3, 1, 0, 2]])
    assert state.miss_count == [0]
    assert state.current_slots == [[3, 1, 0, 2]]
    assert state.lru_slots == [[0, 1, 2, 3]]

    _run(state, [[3, 4, 0, 5]], max_token=64)
    assert state.miss_count == [2]
    assert state.miss_tokens == [[4, 5, -1, -1]]
    assert state.miss_slots == [[1, 2, -1, -1]]
    assert state.slot_to_token == [[0, 4, 5, 3]]
    assert state.lru_slots == [[1, 2, 0, 3]]


def test_plan_duplicate_invalid_future_and_invalid_page() -> None:
    state = PlanState.empty(rows=1, topk=8, capacity=8)
    table = [[0, -1, 2, 3]]

    _run(
        state,
        [[3, 3, -1, 64, 17, 8, 11, 12]],
        visible=[12],
        block_table=table,
        max_token=64,
        block_size=16,
    )

    assert state.miss_count == [4]
    assert state.current_slots == [[0, 1, -1, -1, -1, 2, 3, -1]]
    assert state.miss_tokens == [[3, 3, 8, 11, -1, -1, -1, -1]]


def test_plan_request_reuse_batch_reorder_and_inactive_restoration() -> None:
    state = PlanState.empty(rows=3, topk=2, capacity=2)
    _run(state, [[0, 1], [2, 3], [4, 5]], req_ids=[101, 202, 303])

    _run(
        state,
        [[2, 3], [0, 1]],
        req_ids=[202, 101, 303],
        token_to_req=[0, 1],
    )
    assert state.last_req_ids == [202, 101, 303]
    assert state.miss_count[:2] == [2, 2]
    assert state.last_req_ids[2] == 303
    assert state.slot_to_token[2] == [4, 5]

    _run(
        state,
        [[6, 7], [0, 1], [4, 5]],
        req_ids=[404, 101, 303],
        token_to_req=[0, 1, 2],
    )
    assert state.last_req_ids == [404, 101, 303]
    assert state.slot_to_token[2] == [4, 5]
    assert state.miss_count == [2, 0, 0]


def test_plan_suffix_invalidation_uses_stable_frontier() -> None:
    state = PlanState.empty(rows=1, topk=4, capacity=4)
    _run(state, [[8, 9, 10, 11]])

    _run(state, [[8, 9, 10, 11]], stable=[10])

    assert state.miss_count == [2]
    assert state.current_slots == [[0, 1, 2, 3]]
    assert state.miss_tokens == [[10, 11, -1, -1]]


@pytest.mark.parametrize("width", [1, 4])
def test_real_glm52_topk_and_mtp_shape(width: int) -> None:
    batch = 32
    rows = batch * width
    state = PlanState.empty(rows=rows, topk=2048, capacity=4096)
    topk = [list(range(2048)) for _ in range(rows)]
    token_to_req = [row // width for row in range(rows)]
    visible = [2048 + row % width for row in range(rows)]

    _run(
        state,
        topk,
        req_ids=list(range(10_000, 10_000 + batch)),
        visible=visible,
        token_to_req=token_to_req,
        max_token=143_360,
        block_size=128,
    )

    assert len(state.current_slots) == rows
    assert state.miss_count == [2048] * rows
    assert state.last_req_ids == [10_000 + row // width for row in range(rows)]


def test_workspace_switches_from_ub_to_per_row_gm() -> None:
    assert workspace_elements(128, 2048, 2048) == 1
    assert workspace_elements(128, 2048, 4096) == 1
    assert workspace_elements(128, 2048, 8192) == 3_932_160


def test_transfer_noncontiguous_pages_exact_bits_and_unchanged_slots() -> None:
    state = PlanState.empty(rows=1, topk=2, capacity=4)
    state.miss_count[0] = 2
    state.miss_tokens[0][:2] = [1, 4]
    state.miss_slots[0][:2] = [3, 1]
    host_k = torch.arange(8 * 512, dtype=torch.int32).view(8, 1, 512).to(torch.bfloat16)
    host_rope = (
        torch.arange(8 * 64, dtype=torch.int32).view(8, 1, 64).to(torch.bfloat16)
    )
    resident_k = torch.full((1, 4, 1, 512), -7.0, dtype=torch.bfloat16)
    resident_rope = torch.full((1, 4, 1, 64), -9.0, dtype=torch.bfloat16)

    transfer_reference(
        state=state,
        token_to_req=[0],
        block_table=[[1, 0]],
        active_rows=1,
        block_size=4,
        host_k=host_k,
        host_rope=host_rope,
        resident_k=resident_k,
        resident_rope=resident_rope,
    )

    assert torch.equal(resident_k[0, 3].view(torch.int16), host_k[5].view(torch.int16))
    assert torch.equal(resident_k[0, 1].view(torch.int16), host_k[0].view(torch.int16))
    assert torch.equal(
        resident_rope[0, 3].view(torch.int16), host_rope[5].view(torch.int16)
    )
    assert torch.equal(resident_k[0, 0], torch.full_like(resident_k[0, 0], -7.0))
    assert torch.equal(resident_k[0, 2], torch.full_like(resident_k[0, 2], -7.0))
