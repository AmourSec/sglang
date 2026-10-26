from __future__ import annotations

import pytest
import torch

from .npu_harness import NpuPlanState, call_plan, load_ops

torch_npu = pytest.importorskip("torch_npu")


def _require_npu() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


def _inputs(rows: int, topk: int) -> dict[str, torch.Tensor]:
    return {
        "req_ids": torch.tensor([71, 72], dtype=torch.int64, device="npu"),
        "topk_indices": torch.arange(rows * topk, dtype=torch.int32, device="npu").view(
            rows, topk
        ),
        "stable_prefix_lens": torch.full((rows,), 64, dtype=torch.int32, device="npu"),
        "visible_seq_lens": torch.full((rows,), 64, dtype=torch.int32, device="npu"),
        "token_to_req": torch.arange(rows, dtype=torch.int32, device="npu") % 2,
        "block_table": torch.arange(8, dtype=torch.int32, device="npu").view(2, 4),
        "active_rows": torch.tensor([rows], dtype=torch.int32, device="npu"),
    }


def _call(
    state: NpuPlanState, inputs: dict[str, torch.Tensor], **replacements: torch.Tensor
) -> None:
    call_plan(
        load_ops().plan,
        state,
        **(inputs | replacements),
        topk=8,
        capacity=16,
        max_token=64,
        block_size=16,
        host_num_blocks=8,
    )


def _snapshot(state: NpuPlanState) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.clone()
        for tensor in (
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            state.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
        )
    )


def _assert_snapshot(state: NpuPlanState, snapshot: tuple[torch.Tensor, ...]) -> None:
    actual = (
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
    )
    assert all(torch.equal(left, right) for left, right in zip(actual, snapshot))


def test_binding_rejects_invalid_shape_dtype_layout_and_workspace() -> None:
    _require_npu()
    state = NpuPlanState(4, 8, 16)
    inputs = _inputs(4, 8)
    invalid = (
        {"req_ids": inputs["req_ids"].to(torch.int32)},
        {"topk_indices": inputs["topk_indices"][:, :7]},
        {"topk_indices": inputs["topk_indices"].t()},
        {"active_rows": torch.tensor([4], dtype=torch.int64, device="npu")},
    )
    for replacement in invalid:
        with pytest.raises(RuntimeError):
            _call(state, inputs, **replacement)

    state.workspace = torch.empty(0, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="workspace"):
        _call(state, inputs)


def test_zero_active_rows_and_inactive_rows_preserve_every_state_tensor() -> None:
    _require_npu()
    state = NpuPlanState(4, 8, 16)
    inputs = _inputs(4, 8)
    before = _snapshot(state)
    _call(
        state,
        inputs,
        active_rows=torch.tensor([0], dtype=torch.int32, device="npu"),
    )
    torch_npu.npu.synchronize()
    _assert_snapshot(state, before)

    _call(
        state,
        inputs,
        active_rows=torch.tensor([2], dtype=torch.int32, device="npu"),
    )
    torch_npu.npu.synchronize()
    after = _snapshot(state)
    for old, new in zip(before, after):
        assert torch.equal(old[2:], new[2:])


def test_binding_rejects_unsliced_state_for_a_smaller_call() -> None:
    _require_npu()
    state = NpuPlanState(4, 8, 16)
    inputs = _inputs(3, 8)
    before = _snapshot(state)
    with pytest.raises(RuntimeError, match=r"last_req_ids must have shape \[3\]"):
        load_ops().plan(
            inputs["req_ids"],
            inputs["topk_indices"],
            inputs["stable_prefix_lens"],
            inputs["visible_seq_lens"],
            inputs["token_to_req"],
            inputs["block_table"],
            inputs["active_rows"],
            state.last_req_ids,
            state.slot_to_token[:3],
            state.lru_slots[:3],
            state.current_slots[:3],
            state.miss_count[:3],
            state.miss_tokens[:3],
            state.miss_slots[:3],
            state.workspace,
            8,
            16,
            64,
            16,
            8,
        )
    torch_npu.npu.synchronize()
    _assert_snapshot(state, before)


def test_active_rows_is_clamped_and_invalid_request_mapping_is_empty() -> None:
    _require_npu()
    inputs = _inputs(4, 8)
    clamped = NpuPlanState(4, 8, 16)
    ordinary = NpuPlanState(4, 8, 16)
    _call(
        clamped,
        inputs,
        active_rows=torch.tensor([99], dtype=torch.int32, device="npu"),
    )
    _call(ordinary, inputs)
    torch_npu.npu.synchronize()
    _assert_snapshot(clamped, _snapshot(ordinary))

    invalid = NpuPlanState(4, 8, 16)
    _call(
        invalid,
        inputs,
        token_to_req=torch.tensor([-1, 2, -7, 99], dtype=torch.int32, device="npu"),
    )
    torch_npu.npu.synchronize()
    assert bool((invalid.last_req_ids == -1).all().item())
    assert bool((invalid.slot_to_token == -1).all().item())
    assert bool((invalid.current_slots == -1).all().item())
    assert bool((invalid.miss_count == 0).all().item())


def test_transfer_binding_rejects_host_base_and_resident_contract_violations() -> None:
    _require_npu()
    rows, topk, capacity = 2, 8, 16
    state = NpuPlanState(rows, topk, capacity)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device="npu")
    block_table = torch.arange(4, dtype=torch.int32, device="npu").view(1, 4)
    active_rows = torch.tensor([rows], dtype=torch.int32, device="npu")
    bases = torch.ones(2, dtype=torch.int64, device="npu")
    resident_k = torch.empty(rows, capacity, 1, 512, dtype=torch.bfloat16, device="npu")
    resident_rope = torch.empty(
        rows, capacity, 1, 64, dtype=torch.bfloat16, device="npu"
    )

    def invoke(host_bases: torch.Tensor, k: torch.Tensor, rope: torch.Tensor) -> None:
        load_ops().transfer(
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            token_to_req,
            block_table,
            host_bases,
            active_rows,
            k,
            rope,
            topk,
            capacity,
            16,
            4,
            1024,
            128,
        )

    invalid = (
        (bases.cpu(), resident_k, resident_rope),
        (bases.to(torch.int32), resident_k, resident_rope),
        (bases[:1], resident_k, resident_rope),
        (bases, resident_k.float(), resident_rope),
        (bases, resident_k, resident_rope[:, :, 0]),
    )
    for arguments in invalid:
        with pytest.raises(RuntimeError):
            invoke(*arguments)
