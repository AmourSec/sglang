from __future__ import annotations

import os

import pytest
import torch

from .host_memory import HostPlanes
from .npu_harness import NpuPlanState, assert_state_matches, load_ops
from .reference import PlanState, plan_reference

torch_npu = pytest.importorskip("torch_npu")

BLOCK_SIZE = 128
TOPK = 2048
CAPACITY = 4096
MAX_TOKEN = 143_360
BATCH = 32
HOST_TOKENS = BATCH * MAX_TOKEN


def _require_full_npu() -> None:
    if os.getenv("SGL_SPARSE_KV_TEST_MODE") not in {"full", "host"}:
        pytest.skip("large DDR transfer is selected by the full 950 runner")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


@pytest.fixture(scope="module")
def host_planes() -> HostPlanes:
    _require_full_npu()
    return HostPlanes(HOST_TOKENS)


def _block_table() -> list[list[int]]:
    blocks = MAX_TOKEN // BLOCK_SIZE
    return [
        list(range((request + 1) * blocks - 1, request * blocks - 1, -1))
        for request in range(BATCH)
    ]


def _topk_tokens(visible: int, row: int) -> list[int]:
    tokens = [0, 1, visible // 2, 139_999, visible - 1]
    seen = set(tokens)
    candidate = row % visible
    stride = 67
    while len(tokens) < TOPK:
        if candidate not in seen:
            tokens.append(candidate)
            seen.add(candidate)
        candidate = (candidate + stride) % visible
    return tokens


def _physical_tokens(
    topk_indices: list[list[int]], token_to_req: list[int]
) -> torch.Tensor:
    blocks = MAX_TOKEN // BLOCK_SIZE
    physical: list[list[int]] = []
    for row, tokens in enumerate(topk_indices):
        request = token_to_req[row]
        physical.append(
            [
                (request * blocks + blocks - 1 - token // BLOCK_SIZE) * BLOCK_SIZE
                + token % BLOCK_SIZE
                for token in tokens
            ]
        )
    return torch.tensor(physical, dtype=torch.int64)


def _payload_bits(physical: torch.Tensor, width: int, salt: int) -> torch.Tensor:
    token = physical.to(torch.int64).unsqueeze(-1)
    dimension = torch.arange(width, dtype=torch.int64)
    exponent = ((token + dimension * 131 + salt) % 253 + 1) << 7
    mantissa = (token * 17 + dimension * 29 + salt) % 128
    return (exponent | mantissa).to(torch.int16)


@pytest.mark.parametrize("width", [1, 4])
def test_full_host_plan_transfer_exact_payload_and_guards(
    host_planes: HostPlanes, width: int
) -> None:
    rows = BATCH * width
    visible = [MAX_TOKEN - width + row % width + 1 for row in range(rows)]
    token_to_req = [row // width for row in range(rows)]
    topk_indices = [_topk_tokens(visible[row], row) for row in range(rows)]
    physical = _physical_tokens(topk_indices, token_to_req)
    unique_physical = torch.unique(physical)
    k_bits = _payload_bits(unique_physical, 512, 11)
    rope_bits = _payload_bits(unique_physical, 64, 37)
    host_planes.k[unique_physical] = k_bits.view(torch.bfloat16).unsqueeze(1)
    host_planes.rope[unique_physical] = rope_bits.view(torch.bfloat16).unsqueeze(1)
    table = _block_table()
    req_ids = list(range(50_000, 50_000 + BATCH))
    expected = PlanState.empty(rows, TOPK, CAPACITY)
    plan_reference(
        expected,
        req_ids=req_ids,
        topk_indices=topk_indices,
        stable_prefix_lens=visible,
        visible_seq_lens=visible,
        token_to_req=token_to_req,
        block_table=table,
        active_rows=rows,
        max_token=MAX_TOKEN,
        block_size=BLOCK_SIZE,
        host_num_blocks=BATCH * (MAX_TOKEN // BLOCK_SIZE),
    )
    state = NpuPlanState(rows, TOPK, CAPACITY)
    guard = 4096
    k_payload = rows * CAPACITY * 512
    rope_payload = rows * CAPACITY * 64
    raw_k = torch.full(
        (k_payload + 2 * guard,), -7.0, dtype=torch.bfloat16, device="npu"
    )
    raw_rope = torch.full(
        (rope_payload + 2 * guard,), -9.0, dtype=torch.bfloat16, device="npu"
    )
    resident_k = raw_k[guard:-guard].view(rows, CAPACITY, 1, 512)
    resident_rope = raw_rope[guard:-guard].view(rows, CAPACITY, 1, 64)

    load_ops().plan_transfer(
        torch.tensor(req_ids, dtype=torch.int64, device="npu"),
        torch.tensor(topk_indices, dtype=torch.int32, device="npu"),
        torch.tensor(visible, dtype=torch.int32, device="npu"),
        torch.tensor(visible, dtype=torch.int32, device="npu"),
        torch.tensor(token_to_req, dtype=torch.int32, device="npu"),
        torch.tensor(table, dtype=torch.int32, device="npu"),
        torch.tensor([rows], dtype=torch.int32, device="npu"),
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

    assert_state_matches(state, expected, rows)
    assert bool((raw_k[:guard] == -7).all().item())
    assert bool((raw_k[-guard:] == -7).all().item())
    assert bool((raw_rope[:guard] == -9).all().item())
    assert bool((raw_rope[-guard:] == -9).all().item())
    assert bool((resident_k[:, TOPK:] == -7).all().item())
    assert bool((resident_rope[:, TOPK:] == -9).all().item())
    touched_k_bits = resident_k[:, :TOPK].cpu().view(torch.int16)
    touched_rope_bits = resident_rope[:, :TOPK].cpu().view(torch.int16)
    assert torch.equal(touched_k_bits, _payload_bits(physical, 512, 11).unsqueeze(2))
    assert torch.equal(touched_rope_bits, _payload_bits(physical, 64, 37).unsqueeze(2))
