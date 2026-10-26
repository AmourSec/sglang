from __future__ import annotations

import os

import pytest
import torch

from .host_memory import HostPlanes
from .npu_harness import NpuPlanState, assert_state_matches, load_ops
from .reference import PlanState, plan_reference, transfer_reference

torch_npu = pytest.importorskip("torch_npu")


def _require_graph_npu() -> None:
    if os.getenv("SGL_SPARSE_KV_TEST_MODE") not in {"full", "graph"}:
        pytest.skip("fused graph replay is selected by the 950 graph runner")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def _write_host(host: HostPlanes, version: int) -> None:
    physical = torch.arange(host.k.shape[0], dtype=torch.int32)
    host.k.copy_((version * 1000 + physical).view(-1, 1, 1).to(torch.bfloat16))
    host.rope.copy_((version * 100 + physical).view(-1, 1, 1).to(torch.bfloat16))


def test_fused_graph_replay_observes_metadata_and_real_ddr_payload_changes() -> None:
    _require_graph_npu()
    rows, topk, capacity = 2, 8, 16
    max_token, block_size, host_blocks = 64, 16, 4
    host = HostPlanes(max_token)
    state = NpuPlanState(rows, topk, capacity)
    expected = PlanState.empty(rows, topk, capacity)
    resident_k = torch.full(
        (rows, capacity, 1, 512), -7.0, dtype=torch.bfloat16, device="npu"
    )
    resident_rope = torch.full(
        (rows, capacity, 1, 64), -9.0, dtype=torch.bfloat16, device="npu"
    )
    wanted_k = torch.full((rows, capacity, 1, 512), -7.0, dtype=torch.bfloat16)
    wanted_rope = torch.full((rows, capacity, 1, 64), -9.0, dtype=torch.bfloat16)
    req_ids = torch.tensor([501], dtype=torch.int64, device="npu")
    topk_indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]],
        dtype=torch.int32,
        device="npu",
    )
    stable = torch.full((rows,), 8, dtype=torch.int32, device="npu")
    visible = torch.full((rows,), 8, dtype=torch.int32, device="npu")
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="npu")
    block_table = torch.tensor([[3, 1, 2, 0]], dtype=torch.int32, device="npu")
    active_rows = torch.tensor([rows], dtype=torch.int32, device="npu")
    host_bases = host.bases()
    plan_transfer = load_ops().plan_transfer
    graph = torch.npu.NPUGraph()

    _write_host(host, 0)
    plan_transfer(
        req_ids,
        topk_indices,
        stable,
        visible,
        token_to_req,
        block_table,
        active_rows,
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        state.workspace,
        host_bases,
        resident_k,
        resident_rope,
        topk,
        capacity,
        max_token,
        block_size,
        host_blocks,
        1024,
        128,
    )
    torch_npu.npu.synchronize()
    state.reset()
    resident_k.fill_(-7.0)
    resident_rope.fill_(-9.0)
    torch_npu.npu.synchronize()

    with torch.npu.graph(graph):
        plan_transfer(
            req_ids,
            topk_indices,
            stable,
            visible,
            token_to_req,
            block_table,
            active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            state.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            state.workspace,
            host_bases,
            resident_k,
            resident_rope,
            topk,
            capacity,
            max_token,
            block_size,
            host_blocks,
            1024,
            128,
        )
    torch_npu.npu.synchronize()
    state.reset()
    resident_k.fill_(-7.0)
    resident_rope.fill_(-9.0)

    previous_k = wanted_k.clone()
    changed_counts: list[int] = []
    for version, accepted in ((1, 8), (2, 0), (3, 4), (4, 8)):
        _write_host(host, version)
        stable.fill_(accepted)
        graph.replay()
        torch_npu.npu.synchronize()
        plan_reference(
            expected,
            req_ids=[501],
            topk_indices=[list(range(8)), list(range(7, -1, -1))],
            stable_prefix_lens=[accepted] * rows,
            visible_seq_lens=[8] * rows,
            token_to_req=[0, 0],
            block_table=[[3, 1, 2, 0]],
            active_rows=rows,
            max_token=max_token,
            block_size=block_size,
            host_num_blocks=host_blocks,
        )
        transfer_reference(
            state=expected,
            token_to_req=[0, 0],
            block_table=[[3, 1, 2, 0]],
            active_rows=rows,
            block_size=block_size,
            host_k=host.k,
            host_rope=host.rope,
            resident_k=wanted_k,
            resident_rope=wanted_rope,
        )
        assert_state_matches(state, expected, rows)
        actual_k = resident_k.cpu()
        actual_rope = resident_rope.cpu()
        assert torch.equal(actual_k.view(torch.int16), wanted_k.view(torch.int16))
        assert torch.equal(actual_rope.view(torch.int16), wanted_rope.view(torch.int16))
        changed_counts.append(int((wanted_k != previous_k).any(dim=-1).sum().item()))
        previous_k.copy_(wanted_k)

    assert changed_counts[1] == rows * topk
    assert changed_counts[2] == rows * (topk - 4)
    assert changed_counts[3] == 0
    graph.reset()
    torch_npu.npu.synchronize()
