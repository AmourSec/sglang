from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from .reference import PlanState


class SparseKvOps(Protocol):
    plan: Callable[..., None]
    transfer: Callable[..., None]
    plan_transfer: Callable[..., None]


def hash_capacity(topk: int) -> int:
    capacity = 32
    while capacity < 2 * topk:
        capacity *= 2
    return capacity


def workspace_elements(rows: int, topk: int, capacity: int) -> int:
    row_elements = 3 * hash_capacity(topk) + 2 * capacity + topk
    return 1 if row_elements * 4 <= 112 * 1024 else rows * row_elements


class NpuPlanState:
    def __init__(
        self, rows: int, topk: int, capacity: int, *, device: str = "npu"
    ) -> None:
        self.last_req_ids = torch.full((rows,), -1, dtype=torch.int64, device=device)
        self.slot_to_token = torch.full(
            (rows, capacity), -1, dtype=torch.int32, device=device
        )
        self.lru_slots = torch.arange(
            capacity, dtype=torch.int32, device=device
        ).repeat(rows, 1)
        self.current_slots = torch.full(
            (rows, topk), -1, dtype=torch.int32, device=device
        )
        self.miss_count = torch.zeros(rows, dtype=torch.int32, device=device)
        self.miss_tokens = torch.full(
            (rows, topk), -1, dtype=torch.int32, device=device
        )
        self.miss_slots = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        self.workspace = torch.empty(
            workspace_elements(rows, topk, capacity), dtype=torch.int32, device=device
        )

    def reset(self) -> None:
        rows, capacity = self.slot_to_token.shape
        self.last_req_ids.fill_(-1)
        self.slot_to_token.fill_(-1)
        self.lru_slots.copy_(
            torch.arange(
                capacity, dtype=torch.int32, device=self.lru_slots.device
            ).repeat(rows, 1)
        )
        self.current_slots.fill_(-1)
        self.miss_count.zero_()
        self.miss_tokens.fill_(-1)
        self.miss_slots.fill_(-1)


def load_ops() -> SparseKvOps:
    import sgl_kernel_npu_sparse_kv

    sgl_kernel_npu_sparse_kv.ensure_loaded()
    return torch.ops.sgl_sparse_kv


def call_plan(
    op: Callable[..., None],
    state: NpuPlanState,
    *,
    req_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    stable_prefix_lens: torch.Tensor,
    visible_seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    active_rows: torch.Tensor,
    topk: int,
    capacity: int,
    max_token: int,
    block_size: int,
    host_num_blocks: int,
) -> None:
    rows = topk_indices.shape[0]
    op(
        req_ids,
        topk_indices,
        stable_prefix_lens,
        visible_seq_lens,
        token_to_req,
        block_table,
        active_rows,
        state.last_req_ids[:rows],
        state.slot_to_token[:rows],
        state.lru_slots[:rows],
        state.current_slots[:rows],
        state.miss_count[:rows],
        state.miss_tokens[:rows],
        state.miss_slots[:rows],
        state.workspace,
        topk,
        capacity,
        max_token,
        block_size,
        host_num_blocks,
    )


def assert_state_matches(
    state: NpuPlanState, expected: PlanState, active_rows: int
) -> None:
    actual_tensors = (
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
    )
    expected_values = (
        expected.last_req_ids,
        expected.slot_to_token,
        expected.lru_slots,
        expected.current_slots,
        expected.miss_count,
        expected.miss_tokens,
        expected.miss_slots,
    )
    names = (
        "last_req_ids",
        "slot_to_token",
        "lru_slots",
        "current_slots",
        "miss_count",
        "miss_tokens",
        "miss_slots",
    )
    for name, actual, wanted in zip(names, actual_tensors, expected_values):
        actual_cpu = actual[:active_rows].cpu()
        wanted_cpu = torch.tensor(wanted[:active_rows], dtype=actual_cpu.dtype)
        assert torch.equal(actual_cpu, wanted_cpu), name
