from __future__ import annotations

import os

import pytest
import torch

from .host_memory import HostPlanes
from .npu_harness import NpuPlanState, assert_state_matches, load_ops
from .reference import PlanState, plan_reference
from .test_npu_full_host import (
    BATCH,
    BLOCK_SIZE,
    CAPACITY,
    HOST_TOKENS,
    MAX_TOKEN,
    TOPK,
    _block_table,
    _payload_bits,
)

torch_npu = pytest.importorskip("torch_npu")


def _require_host_npu() -> None:
    if os.getenv("SGL_SPARSE_KV_TEST_MODE") not in {"full", "host"}:
        pytest.skip("mixed host matrix is selected by the 950 host runner")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


@pytest.fixture(scope="module")
def host_planes() -> HostPlanes:
    _require_host_npu()
    return HostPlanes(HOST_TOKENS)


def _tokens(visible: int, row: int) -> list[int]:
    valid_count = min(visible, TOPK)
    valid: list[int] = []
    seen: set[int] = set()
    candidate = row
    while len(valid) < valid_count:
        token = candidate % visible
        if token not in seen:
            valid.append(token)
            seen.add(token)
        candidate += 67
    return valid + [visible + offset for offset in range(TOPK - valid_count)]


def _physical(request: int, token: int) -> int:
    blocks = MAX_TOKEN // BLOCK_SIZE
    block = request * blocks + blocks - 1 - token // BLOCK_SIZE
    return block * BLOCK_SIZE + token % BLOCK_SIZE


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 31])
def test_mixed_sequences_and_batch_sizes_use_exact_host_addresses(
    host_planes: HostPlanes, batch: int
) -> None:
    visible_cases = [1, 127, 128, 129, 2048, 32_768, 140_000, MAX_TOKEN]
    visible = [visible_cases[row % len(visible_cases)] for row in range(batch)]
    topk_indices = [_tokens(visible[row], row) for row in range(batch)]
    valid_physical = torch.tensor(
        [
            _physical(row, token)
            for row in range(batch)
            for token in topk_indices[row]
            if token < visible[row]
        ],
        dtype=torch.int64,
    )
    unique_physical = torch.unique(valid_physical)
    host_planes.k[unique_physical] = (
        _payload_bits(unique_physical, 512, 11).view(torch.bfloat16).unsqueeze(1)
    )
    host_planes.rope[unique_physical] = (
        _payload_bits(unique_physical, 64, 37).view(torch.bfloat16).unsqueeze(1)
    )
    state = NpuPlanState(batch, TOPK, CAPACITY)
    expected = PlanState.empty(batch, TOPK, CAPACITY)
    request_ids = list(range(70_000, 70_000 + BATCH))
    table = _block_table()
    plan_reference(
        expected,
        req_ids=request_ids,
        topk_indices=topk_indices,
        stable_prefix_lens=visible,
        visible_seq_lens=visible,
        token_to_req=list(range(batch)),
        block_table=table,
        active_rows=batch,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=BATCH * (MAX_TOKEN // BLOCK_SIZE),
    )
    resident_k = torch.full(
        (batch, CAPACITY, 1, 512), -7.0, dtype=torch.bfloat16, device="npu"
    )
    resident_rope = torch.full(
        (batch, CAPACITY, 1, 64), -9.0, dtype=torch.bfloat16, device="npu"
    )
    load_ops().plan_transfer(
        torch.tensor(request_ids, dtype=torch.int64, device="npu"),
        torch.tensor(topk_indices, dtype=torch.int32, device="npu"),
        torch.tensor(visible, dtype=torch.int32, device="npu"),
        torch.tensor(visible, dtype=torch.int32, device="npu"),
        torch.arange(batch, dtype=torch.int32, device="npu"),
        torch.tensor(table, dtype=torch.int32, device="npu"),
        torch.tensor([batch], dtype=torch.int32, device="npu"),
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        state.workspace,
        host_planes.bases(),
        resident_k,
        resident_rope,
        TOPK,
        CAPACITY,
        MAX_TOKEN,
        BLOCK_SIZE,
        BATCH * (MAX_TOKEN // BLOCK_SIZE),
        1024,
        128,
    )
    torch_npu.npu.synchronize()
    assert_state_matches(state, expected, batch)
    actual_k = resident_k.cpu().view(torch.int16)
    actual_rope = resident_rope.cpu().view(torch.int16)
    for row in range(batch):
        touched: set[int] = set()
        for position, token in enumerate(topk_indices[row]):
            slot = expected.current_slots[row][position]
            if slot < 0:
                continue
            touched.add(slot)
            physical = torch.tensor([_physical(row, token)], dtype=torch.int64)
            assert torch.equal(
                actual_k[row, slot], _payload_bits(physical, 512, 11).view(1, 1, 512)[0]
            )
            assert torch.equal(
                actual_rope[row, slot],
                _payload_bits(physical, 64, 37).view(1, 1, 64)[0],
            )
        untouched = [slot for slot in range(CAPACITY) if slot not in touched]
        assert bool((resident_k[row, untouched] == -7).all().item())
        assert bool((resident_rope[row, untouched] == -9).all().item())
