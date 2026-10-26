from __future__ import annotations

from typing import Callable

import pytest
import torch

from .perf.cases import TOPK, BenchmarkCase
from .perf.workload import build_workload
from .reference import PlanState, plan_reference, transfer_reference


class CpuHost:
    def __init__(self, tokens: int) -> None:
        self.k = torch.empty((tokens, 1, 512), dtype=torch.bfloat16)
        self.rope = torch.empty((tokens, 1, 64), dtype=torch.bfloat16)

    def bases(self) -> torch.Tensor:
        return torch.tensor([self.k.data_ptr(), self.rope.data_ptr()])


class CpuOps:
    def __init__(self, host: CpuHost) -> None:
        self.host = host
        self.plan_calls = 0

    def plan(self, *args: torch.Tensor | int) -> None:
        self.plan_calls += 1
        inputs = args[:7]
        buffers = args[7:14]
        topk, capacity, max_token, block_size, host_blocks = args[15:]
        state = PlanState.empty(len(buffers[0]), topk, capacity)
        names = tuple(vars(state))
        for name, buffer in zip(names, buffers):
            setattr(state, name, buffer.tolist())
        req, query, stable, visible, mapping, table, active = inputs
        plan_reference(
            state,
            req_ids=req.tolist(),
            topk_indices=query.tolist(),
            stable_prefix_lens=stable.tolist(),
            visible_seq_lens=visible.tolist(),
            token_to_req=mapping.tolist(),
            block_table=table.tolist(),
            active_rows=int(active[0]),
            max_token=max_token,
            block_size=block_size,
            host_num_blocks=host_blocks,
        )
        for name, buffer in zip(names, buffers):
            buffer.copy_(torch.tensor(getattr(state, name), dtype=buffer.dtype))

    def transfer(self, *args: torch.Tensor | int) -> None:
        count, tokens, slots, mapping, table, _, active, k, rope = args[:9]
        topk, capacity, block_size, _, k_bytes, rope_bytes = args[9:]
        assert (k_bytes, rope_bytes) == (1024, 128)
        state = PlanState.empty(len(count), topk, capacity)
        state.miss_count = count.tolist()
        state.miss_tokens = tokens.tolist()
        state.miss_slots = slots.tolist()
        transfer_reference(
            state=state,
            token_to_req=mapping.tolist(),
            block_table=table.tolist(),
            active_rows=int(active[0]),
            block_size=block_size,
            host_k=self.host.k,
            host_rope=self.host.rope,
            resident_k=k,
            resident_rope=rope,
        )

    def plan_transfer(self, *args: torch.Tensor | int) -> None:
        self.plan(*args[:15], *args[18:23])
        self.transfer(
            *args[11:14],
            args[4],
            args[5],
            args[15],
            args[6],
            args[16],
            args[17],
            *args[18:20],
            *args[21:],
        )


@pytest.mark.parametrize("mtp,hit_ratio", [(0, 0.0), (1, 0.25), (2, 1.0), (3, 0.5)])
def test_device_calls_preserve_hit_ratio_and_transfer_is_independent(
    mtp: int,
    hit_ratio: float,
) -> None:
    from .perf.device import DeviceWorkload

    case = BenchmarkCase(1, mtp, 4096, hit_ratio, capacity=2048)
    host = CpuHost(case.host_tokens)
    ops = CpuOps(host)
    device = DeviceWorkload(build_workload(case), host, ops, device="cpu")
    for operation in ("plan", "transfer", "plan_transfer"):
        call: Callable[[], None] = device.calls[operation]
        device.prepare(operation)
        call()
        device.check(operation)
        plan_before = ops.plan_calls
        for _ in range(2):
            device.restore(operation)
            call()
            device.check(operation)
        if operation == "transfer":
            assert ops.plan_calls == plan_before
        assert device.state.miss_count.tolist() == [case.misses_per_row] * case.rows
    device.restore("plan")
    assert device.state.miss_count.tolist() == [0] * case.rows
    assert device.state.current_slots.eq(-1).all()
    assert device.state.miss_tokens.eq(-1).all()
    assert device.state.miss_slots.eq(-1).all()
    assert device.state.slot_to_token.shape == (case.rows, case.capacity)
    assert device.state.current_slots.shape == (case.rows, TOPK)


def test_payload_check_detects_wrong_source() -> None:
    from .perf.device import DeviceWorkload

    case = BenchmarkCase(1, 0, 4096, 0.0, capacity=2048)
    host = CpuHost(case.host_tokens)
    device = DeviceWorkload(build_workload(case), host, CpuOps(host), device="cpu")
    device.prepare("transfer")
    device.calls["transfer"]()
    device.check("transfer")
    device.resident_k.zero_()
    with pytest.raises(AssertionError):
        device.check("transfer")
