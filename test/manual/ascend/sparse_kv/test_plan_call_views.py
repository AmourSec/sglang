from __future__ import annotations

import pytest
import torch

from .npu_harness import NpuPlanState, call_plan, workspace_elements


def _state_tensors(state: NpuPlanState) -> tuple[torch.Tensor, ...]:
    return (
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
    )


@pytest.mark.parametrize("capacity", [4096, 8192])
@pytest.mark.parametrize("call_rows", [3, 4])
def test_plan_call_reuses_storage_across_shrink_and_restore(
    capacity: int, call_rows: int
) -> None:
    topk = 2048
    state = NpuPlanState(4, topk, capacity, device="cpu")
    buffers = _state_tensors(state)
    for index, buffer in enumerate(buffers):
        buffer.fill_(100 + index)
    before = tuple(buffer.clone() for buffer in buffers)
    calls: list[int] = []

    def check_views(
        req_ids: torch.Tensor,
        indices: torch.Tensor,
        stable: torch.Tensor,
        visible: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        active: torch.Tensor,
        last_req_ids: torch.Tensor,
        slot_to_token: torch.Tensor,
        lru_slots: torch.Tensor,
        current_slots: torch.Tensor,
        miss_count: torch.Tensor,
        miss_tokens: torch.Tensor,
        miss_slots: torch.Tensor,
        workspace: torch.Tensor,
        *attributes: int,
    ) -> None:
        rows = indices.shape[0]
        views = (
            last_req_ids,
            slot_to_token,
            lru_slots,
            current_slots,
            miss_count,
            miss_tokens,
            miss_slots,
        )
        assert req_ids.shape == (4,)
        assert block_table.shape[0] == 4
        assert stable.shape == visible.shape == token_to_req.shape == (rows,)
        assert attributes[:2] == (topk, capacity)
        assert workspace.data_ptr() == state.workspace.data_ptr()
        assert workspace.numel() >= workspace_elements(rows, topk, capacity)
        for index, (view, buffer) in enumerate(zip(views, buffers)):
            assert view.shape == (rows, *buffer.shape[1:]), "per-call row mismatch"
            assert view.is_contiguous()
            assert view.data_ptr() == buffer.data_ptr(), "state was reallocated"
            if calls:
                assert torch.equal(view[3], before[index][3])
            view[: int(active.item())].fill_(200 + len(calls))
        calls.append(rows)

    def invoke(rows: int, active_rows: int) -> None:
        call_plan(
            check_views,
            state,
            req_ids=torch.arange(4, dtype=torch.int64),
            topk_indices=torch.zeros(rows, topk, dtype=torch.int32),
            stable_prefix_lens=torch.full((rows,), 32, dtype=torch.int32),
            visible_seq_lens=torch.full((rows,), 32, dtype=torch.int32),
            token_to_req=torch.arange(rows, dtype=torch.int32),
            block_table=torch.arange(4, dtype=torch.int32).view(4, 1),
            active_rows=torch.tensor([active_rows], dtype=torch.int32),
            topk=topk,
            capacity=capacity,
            max_token=32,
            block_size=128,
            host_num_blocks=4,
        )

    invoke(call_rows, 3)
    for buffer, original in zip(buffers, before):
        assert torch.equal(buffer[3:], original[3:]), "inactive row was modified"
        assert bool(buffer[:3].eq(200).all())
    invoke(4, 4)
    assert calls == [call_rows, 4]
    assert all(bool(buffer.eq(201).all()) for buffer in buffers)
