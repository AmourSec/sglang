from __future__ import annotations

import math

import pytest
import torch

from .npu_harness import NpuPlanState, call_plan, load_ops

torch_npu = pytest.importorskip("torch_npu")

TOPK = 2048
CAPACITY = 4096
BLOCK_SIZE = 128
MAX_TOKEN = 143_360
HEADS = 64


class _Resident:
    pass


class _Manager:
    def prepare_resident(
        self, layer: object, batch: object, topk: torch.Tensor
    ) -> object:
        return self.resident


class _Backend:
    pass


class _Layer:
    pass


class _Batch:
    pass


def _require_npu() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def _selected_tokens(visible: int, row: int) -> list[int]:
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


def _build_adapter_inputs(width: int, visible: int) -> tuple[object, ...]:
    rows = width
    visible_lens = [visible - width + row + 1 for row in range(rows)]
    tokens = [_selected_tokens(visible_lens[row], row) for row in range(rows)]
    state = NpuPlanState(rows, TOPK, CAPACITY)
    call_plan(
        load_ops().plan,
        state,
        req_ids=torch.tensor([9001], dtype=torch.int64, device="npu"),
        topk_indices=torch.tensor(tokens, dtype=torch.int32, device="npu"),
        stable_prefix_lens=torch.tensor(visible_lens, dtype=torch.int32, device="npu"),
        visible_seq_lens=torch.tensor(visible_lens, dtype=torch.int32, device="npu"),
        token_to_req=torch.zeros(rows, dtype=torch.int32, device="npu"),
        block_table=torch.arange(
            MAX_TOKEN // BLOCK_SIZE, dtype=torch.int32, device="npu"
        ).view(1, -1),
        active_rows=torch.tensor([rows], dtype=torch.int32, device="npu"),
        topk=TOPK,
        capacity=CAPACITY,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=MAX_TOKEN // BLOCK_SIZE,
    )
    torch_npu.npu.synchronize()
    generator = torch.Generator().manual_seed(700 + width + visible)
    resident_k_cpu = (
        torch.randn(rows, CAPACITY, 1, 512, generator=generator, dtype=torch.float32)
        .mul_(0.03)
        .to(torch.bfloat16)
    )
    resident_rope_cpu = (
        torch.randn(rows, CAPACITY, 1, 64, generator=generator, dtype=torch.float32)
        .mul_(0.03)
        .to(torch.bfloat16)
    )
    resident = _Resident()
    resident.current_slots = state.current_slots
    resident.row_valid = state.current_slots.ge(0).any(dim=1)
    resident.k = resident_k_cpu.to("npu")
    resident.rope = resident_rope_cpu.to("npu")
    manager = _Manager()
    manager.capacity = CAPACITY
    manager.block_size = BLOCK_SIZE
    manager.resident = resident
    pages = CAPACITY // BLOCK_SIZE
    manager.resident_block_table = torch.arange(
        rows * pages, dtype=torch.int32, device="npu"
    ).view(rows, pages)
    manager.resident_query_lens = torch.arange(
        1, rows + 1, dtype=torch.int32, device="npu"
    )
    manager.visible_seq_lens = torch.tensor(
        visible_lens, dtype=torch.int32, device="npu"
    )
    backend = _Backend()
    backend.sparse_kv_manager = manager
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    layer = _Layer()
    layer.tp_q_head_num = HEADS
    layer.scaling = 1.0 / math.sqrt(576)
    query = torch.randn(rows, HEADS, 512, generator=generator).mul_(0.03)
    query_rope = torch.randn(rows, HEADS, 64, generator=generator).mul_(0.03)
    return backend, layer, tokens, resident_k_cpu, resident_rope_cpu, query, query_rope


def _attention_oracle(
    slots: torch.Tensor,
    resident_k: torch.Tensor,
    resident_rope: torch.Tensor,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    rows = slots.shape[0]
    output = torch.zeros(rows, HEADS, 512, dtype=torch.float32)
    for row in range(rows):
        valid_slots = slots[row][slots[row] >= 0].to(torch.int64).cpu()
        if valid_slots.numel() == 0:
            continue
        keys = resident_k[row, valid_slots, 0].float()
        ropes = resident_rope[row, valid_slots, 0].float()
        scores = query[row].float() @ keys.t()
        scores.add_(query_rope[row].float() @ ropes.t()).mul_(scale)
        output[row] = torch.softmax(scores, dim=-1) @ keys
    return output


@pytest.mark.parametrize(
    ("width", "visible"), [(1, 33), (4, 33), (1, 140_000), (4, 140_000)]
)
def test_runtime_attention_matches_fp32_selected_kv_oracle(
    width: int, visible: int
) -> None:
    _require_npu()
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_attention import (
        forward_sparse_kv_simt,
    )

    backend, layer, tokens, resident_k, resident_rope, query, query_rope = (
        _build_adapter_inputs(width, visible)
    )
    slots = backend.sparse_kv_manager.resident.current_slots
    for row in range(width):
        valid_count = min(visible - width + row + 1, TOPK)
        assert bool((slots[row, :valid_count] >= 0).all().item())
        assert bool((slots[row, valid_count:] == -1).all().item())
    wanted = _attention_oracle(
        slots, resident_k, resident_rope, query, query_rope, layer.scaling
    )
    actual = forward_sparse_kv_simt(
        backend,
        query.to(torch.bfloat16).to("npu"),
        query_rope.to(torch.bfloat16).to("npu"),
        layer,
        _Batch(),
        torch.tensor(tokens, dtype=torch.int32, device="npu"),
    )
    torch_npu.npu.synchronize()
    torch.testing.assert_close(
        actual.cpu().float().view(width, HEADS, 512), wanted, rtol=0.06, atol=0.02
    )


def test_runtime_attention_graph_padding_is_finite_zero() -> None:
    _require_npu()
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_attention import (
        forward_sparse_kv_simt,
    )

    backend, layer, _, _, _, query, query_rope = _build_adapter_inputs(1, 1)
    resident = backend.sparse_kv_manager.resident
    padding_state = NpuPlanState(1, TOPK, CAPACITY)
    future_topk = torch.full((1, TOPK), 7, dtype=torch.int32, device="npu")
    call_plan(
        load_ops().plan,
        padding_state,
        req_ids=torch.tensor([9001], dtype=torch.int64, device="npu"),
        topk_indices=future_topk,
        stable_prefix_lens=torch.tensor([1], dtype=torch.int32, device="npu"),
        visible_seq_lens=torch.tensor([1], dtype=torch.int32, device="npu"),
        token_to_req=torch.zeros(1, dtype=torch.int32, device="npu"),
        block_table=torch.arange(
            MAX_TOKEN // BLOCK_SIZE, dtype=torch.int32, device="npu"
        ).view(1, -1),
        active_rows=torch.tensor([1], dtype=torch.int32, device="npu"),
        topk=TOPK,
        capacity=CAPACITY,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=MAX_TOKEN // BLOCK_SIZE,
    )
    torch_npu.npu.synchronize()
    resident.current_slots = padding_state.current_slots
    resident.row_valid = padding_state.current_slots.ge(0).any(dim=1)
    assert bool((resident.current_slots == -1).all().item())
    q_npu = query.to(torch.bfloat16).to("npu")
    q_rope_npu = query_rope.to(torch.bfloat16).to("npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = forward_sparse_kv_simt(
            backend,
            q_npu,
            q_rope_npu,
            layer,
            _Batch(),
            future_topk,
        )
    graph.replay()
    torch_npu.npu.synchronize()
    assert bool(torch.isfinite(result).all().item())
    assert bool((result == 0).all().item())
    graph.reset()
