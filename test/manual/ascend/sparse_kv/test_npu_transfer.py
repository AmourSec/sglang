from __future__ import annotations

import pytest
import torch

from .host_memory import HostPlanes
from .npu_harness import NpuPlanState, assert_state_matches, load_ops
from .reference import PlanState, plan_reference, transfer_reference

torch_npu = pytest.importorskip("torch_npu")


def _require_npu() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def test_transfer_reads_real_ddr_dva_exact_bits_and_preserves_guards() -> None:
    _require_npu()
    block_size = 4
    rows = 2
    topk = 4
    capacity = 6
    host = HostPlanes(tokens=16)
    host.k.copy_(
        torch.arange(host.k.numel(), dtype=torch.int32)
        .view_as(host.k)
        .to(torch.bfloat16)
    )
    host.rope.copy_(
        torch.arange(host.rope.numel(), dtype=torch.int32)
        .view_as(host.rope)
        .to(torch.bfloat16)
    )
    expected = PlanState.empty(rows, topk, capacity)
    expected.miss_count[:] = [2, 2]
    expected.miss_tokens[0][:2] = [1, 6]
    expected.miss_tokens[1][:2] = [2, 7]
    expected.miss_slots[0][:2] = [4, 1]
    expected.miss_slots[1][:2] = [3, 5]
    actual = NpuPlanState(rows, topk, capacity)
    actual.miss_count.copy_(
        torch.tensor(expected.miss_count, dtype=torch.int32, device="npu")
    )
    actual.miss_tokens.copy_(
        torch.tensor(expected.miss_tokens, dtype=torch.int32, device="npu")
    )
    actual.miss_slots.copy_(
        torch.tensor(expected.miss_slots, dtype=torch.int32, device="npu")
    )
    token_to_req = [0, 1]
    block_table = [[2, 0], [3, 1]]
    resident_k = torch.full(
        (rows, capacity, 1, 512), -7.0, dtype=torch.bfloat16, device="npu"
    )
    resident_rope = torch.full(
        (rows, capacity, 1, 64), -9.0, dtype=torch.bfloat16, device="npu"
    )
    wanted_k = torch.full((rows, capacity, 1, 512), -7.0, dtype=torch.bfloat16)
    wanted_rope = torch.full((rows, capacity, 1, 64), -9.0, dtype=torch.bfloat16)
    transfer_reference(
        state=expected,
        token_to_req=token_to_req,
        block_table=block_table,
        active_rows=rows,
        block_size=block_size,
        host_k=host.k,
        host_rope=host.rope,
        resident_k=wanted_k,
        resident_rope=wanted_rope,
    )

    load_ops().transfer(
        actual.miss_count,
        actual.miss_tokens,
        actual.miss_slots,
        torch.tensor(token_to_req, dtype=torch.int32, device="npu"),
        torch.tensor(block_table, dtype=torch.int32, device="npu"),
        host.bases(),
        torch.tensor([rows], dtype=torch.int32, device="npu"),
        resident_k,
        resident_rope,
        topk,
        capacity,
        block_size,
        4,
        512 * torch.bfloat16.itemsize,
        64 * torch.bfloat16.itemsize,
    )
    torch_npu.npu.synchronize()

    actual_k = resident_k.cpu()
    actual_rope = resident_rope.cpu()
    assert torch.equal(actual_k.view(torch.int16), wanted_k.view(torch.int16))
    assert torch.equal(actual_rope.view(torch.int16), wanted_rope.view(torch.int16))

    fused_state = NpuPlanState(rows, topk, capacity)
    fused_expected = PlanState.empty(rows, topk, capacity)
    fused_topk = [[1, 6, 0, 5], [2, 7, 3, 4]]
    req_ids = [91, 92]
    stable = [8, 8]
    visible = [8, 8]
    plan_reference(
        fused_expected,
        req_ids=req_ids,
        topk_indices=fused_topk,
        stable_prefix_lens=stable,
        visible_seq_lens=visible,
        token_to_req=token_to_req,
        block_table=block_table,
        active_rows=rows,
        max_token=8,
        block_size=block_size,
        host_num_blocks=4,
    )
    resident_k.fill_(-7.0)
    resident_rope.fill_(-9.0)
    wanted_k.fill_(-7.0)
    wanted_rope.fill_(-9.0)
    transfer_reference(
        state=fused_expected,
        token_to_req=token_to_req,
        block_table=block_table,
        active_rows=rows,
        block_size=block_size,
        host_k=host.k,
        host_rope=host.rope,
        resident_k=wanted_k,
        resident_rope=wanted_rope,
    )
    load_ops().plan_transfer(
        torch.tensor(req_ids, dtype=torch.int64, device="npu"),
        torch.tensor(fused_topk, dtype=torch.int32, device="npu"),
        torch.tensor(stable, dtype=torch.int32, device="npu"),
        torch.tensor(visible, dtype=torch.int32, device="npu"),
        torch.tensor(token_to_req, dtype=torch.int32, device="npu"),
        torch.tensor(block_table, dtype=torch.int32, device="npu"),
        torch.tensor([rows], dtype=torch.int32, device="npu"),
        fused_state.last_req_ids,
        fused_state.slot_to_token,
        fused_state.lru_slots,
        fused_state.current_slots,
        fused_state.miss_count,
        fused_state.miss_tokens,
        fused_state.miss_slots,
        fused_state.workspace,
        host.bases(),
        resident_k,
        resident_rope,
        topk,
        capacity,
        8,
        block_size,
        4,
        512 * torch.bfloat16.itemsize,
        64 * torch.bfloat16.itemsize,
    )
    torch_npu.npu.synchronize()
    assert_state_matches(fused_state, fused_expected, rows)
    assert torch.equal(resident_k.cpu().view(torch.int16), wanted_k.view(torch.int16))
    assert torch.equal(
        resident_rope.cpu().view(torch.int16), wanted_rope.view(torch.int16)
    )
