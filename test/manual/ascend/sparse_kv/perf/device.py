from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol

import torch

from ..npu_harness import NpuPlanState, SparseKvOps, assert_state_matches
from .cases import BLOCK_SIZE, TOPK
from .workload import Workload, oracle_post_state


class HostMemory(Protocol):
    k: torch.Tensor
    rope: torch.Tensor

    def bases(self) -> torch.Tensor: ...


def _payload(physical: torch.Tensor, width: int, salt: int) -> torch.Tensor:
    token = physical.to(torch.int64).unsqueeze(-1)
    dimension = torch.arange(width, dtype=torch.int64)
    exponent = ((token + dimension * 131 + salt) % 253 + 1) << 7
    mantissa = (token * 17 + dimension * 29 + salt) % 128
    return (exponent | mantissa).to(torch.int16).view(torch.bfloat16).unsqueeze(1)


def _physical(workload: Workload, row: int, token: int) -> int:
    request = workload.token_to_req[row]
    return (
        workload.block_table[request][token // BLOCK_SIZE] * BLOCK_SIZE
        + token % BLOCK_SIZE
    )


def _populate_host(workload: Workload, host: HostMemory) -> None:
    physical = torch.tensor(
        sorted(
            {
                _physical(workload, row, token)
                for row, query in enumerate(workload.topk_indices)
                for token in query
            }
        ),
        dtype=torch.int64,
    )
    # Only queried DDR addresses are read; bound temporary payload allocations.
    for chunk in physical.split(4096):
        host.k[chunk] = _payload(chunk, 512, 11)
        host.rope[chunk] = _payload(chunk, 64, 37)


class DeviceWorkload:
    def __init__(
        self,
        workload: Workload,
        host: HostMemory,
        ops: SparseKvOps,
        *,
        device: str = "npu",
    ) -> None:
        self.workload = workload
        self.expected = oracle_post_state(workload)
        self.host = host
        case = workload.case
        if self.expected.miss_count != [case.misses_per_row] * case.rows:
            raise ValueError("workload does not produce the requested hit ratio")
        self.state = NpuPlanState(case.rows, TOPK, case.capacity, device=device)
        self.buffers = tuple(getattr(self.state, name) for name in vars(self.expected))
        self.seed = tuple(
            torch.tensor(
                getattr(workload.initial_state, name), dtype=buffer.dtype, device=device
            )
            for name, buffer in zip(vars(self.expected), self.buffers)
        )
        inputs = (
            torch.tensor(workload.req_ids, dtype=torch.int64, device=device),
            torch.tensor(workload.topk_indices, dtype=torch.int32, device=device),
            torch.tensor(workload.stable_prefix_lens, dtype=torch.int32, device=device),
            torch.tensor(workload.visible_seq_lens, dtype=torch.int32, device=device),
            torch.tensor(workload.token_to_req, dtype=torch.int32, device=device),
            torch.tensor(workload.block_table, dtype=torch.int32, device=device),
            torch.tensor([case.rows], dtype=torch.int32, device=device),
        )
        self.resident_k = torch.empty(
            (case.rows, case.capacity, 1, 512), dtype=torch.bfloat16, device=device
        )
        self.resident_rope = torch.empty(
            (case.rows, case.capacity, 1, 64), dtype=torch.bfloat16, device=device
        )
        host_bases = host.bases()
        plan_inputs = (*inputs, *self.buffers, self.state.workspace)
        attributes = (
            TOPK,
            case.capacity,
            case.seq_len + case.width,
            BLOCK_SIZE,
            case.host_num_blocks,
        )
        self.calls: dict[str, Callable[[], None]] = {
            "plan": partial(ops.plan, *plan_inputs, *attributes),
            "transfer": partial(
                ops.transfer,
                self.state.miss_count,
                self.state.miss_tokens,
                self.state.miss_slots,
                inputs[4],
                inputs[5],
                host_bases,
                inputs[6],
                self.resident_k,
                self.resident_rope,
                TOPK,
                case.capacity,
                BLOCK_SIZE,
                case.host_num_blocks,
                1024,
                128,
            ),
            "plan_transfer": partial(
                ops.plan_transfer,
                *plan_inputs,
                host_bases,
                self.resident_k,
                self.resident_rope,
                *attributes,
                1024,
                128,
            ),
        }
        _populate_host(workload, host)
        self.restore("plan")
        self.calls["plan"]()
        self.check("plan")
        self.post_plan = tuple(buffer.clone() for buffer in self.buffers)
        self._prepare_payload_checks(device)

    def restore(self, operation: str) -> None:
        source = self.post_plan if operation == "transfer" else self.seed
        for buffer, snapshot in zip(self.buffers, source):
            buffer.copy_(snapshot)

    def prepare(self, operation: str) -> None:
        self.restore(operation)
        self.resident_k.fill_(-7.0)
        self.resident_rope.fill_(-9.0)

    def _prepare_payload_checks(self, device: str) -> None:
        rows, slots, physical = [], [], []
        untouched_rows, untouched_slots = [], []
        case = self.workload.case
        for row in range(case.rows):
            count = self.expected.miss_count[row]
            selected = (
                sorted({index * (count - 1) // 7 for index in range(8)})
                if count
                else []
            )
            for index in selected:
                rows.append(row)
                slots.append(self.expected.miss_slots[row][index])
                physical.append(
                    _physical(self.workload, row, self.expected.miss_tokens[row][index])
                )
            written = set(self.expected.miss_slots[row][:count])
            untouched = [slot for slot in range(case.capacity) if slot not in written]
            for slot in untouched[:: max(1, len(untouched) // 8)][:8]:
                untouched_rows.append(row)
                untouched_slots.append(slot)
        self.sample_index = (
            torch.tensor(rows, dtype=torch.int64, device=device),
            torch.tensor(slots, dtype=torch.int64, device=device),
        )
        self.untouched_index = (
            torch.tensor(untouched_rows, dtype=torch.int64, device=device),
            torch.tensor(untouched_slots, dtype=torch.int64, device=device),
        )
        host_index = torch.tensor(physical, dtype=torch.int64)
        self.wanted_k = self.host.k[host_index].clone().view(torch.int16)
        self.wanted_rope = self.host.rope[host_index].clone().view(torch.int16)

    def check(self, operation: str) -> None:
        assert_state_matches(self.state, self.expected, self.workload.case.rows)
        if operation == "plan":
            return
        for resident, wanted, sentinel in (
            (self.resident_k, self.wanted_k, -7),
            (self.resident_rope, self.wanted_rope, -9),
        ):
            actual = resident[self.sample_index].cpu().view(torch.int16)
            assert torch.equal(actual, wanted), (
                "sampled Transfer payload differs from host DDR"
            )
            assert bool(resident[self.untouched_index].eq(sentinel).all().item()), (
                "Transfer modified a sampled non-miss slot"
            )
