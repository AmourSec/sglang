from __future__ import annotations

import pytest
import torch

from .runtime_harness import (
    Batch,
    RuntimeContext,
    assert_real_dva_and_factory_sharing,
    make_context,
    make_draft,
    payload,
    topk,
)

torch_npu = pytest.importorskip("torch_npu")

TensorPair = tuple[torch.Tensor, torch.Tensor]


def _assert_prefill_and_decode(
    context: RuntimeContext,
) -> tuple[TensorPair, TensorPair]:
    stream = torch.npu.current_stream()
    chunks: list[TensorPair] = []
    for start, rows, seed in ((0, 3, 11), (3, 2, 29)):
        batch = Batch(
            context.extend_mode,
            context.req,
            torch.arange(start, start + rows, device="npu"),
            context.active[:rows],
            extend_seq_lens=torch.tensor([rows], device="npu"),
        )
        chunk = (payload(rows, 512, seed), payload(rows, 64, seed + 3))
        context.target.offload_v2(*chunk, context.layer_8, batch, stream)
        chunks.append(chunk)
    expected = (
        torch.cat([part[0] for part in chunks]),
        torch.cat([part[1] for part in chunks]),
    )
    layer_7_payload = (payload(5, 512, 101), payload(5, 64, 107))
    layer_7_batch = Batch(
        context.extend_mode,
        context.req,
        torch.arange(5, device="npu"),
        context.active[:5],
        extend_seq_lens=torch.tensor([5], device="npu"),
    )
    context.target.offload_v2(*layer_7_payload, context.layer_7, layer_7_batch, stream)
    for position, seed in ((5, 47), (6, 59)):
        decode = Batch(
            context.decode_mode,
            context.req,
            torch.tensor([position], device="npu"),
            context.active[:1],
        )
        chunk = (payload(1, 512, seed), payload(1, 64, seed + 3))
        context.target.offload_v2(*chunk, context.layer_8, decode, stream)
        expected = (
            torch.cat((expected[0], chunk[0])),
            torch.cat((expected[1], chunk[1])),
        )
    gather = Batch(
        context.decode_mode,
        context.req,
        torch.tensor([6], device="npu"),
        context.active[:1],
        seq_lens=torch.tensor([7], device="npu"),
    )
    layer_gather = gather._replace(seq_lens=torch.tensor([5], device="npu"))
    actual = context.target.get_forward_kv(context.layer_8, gather, stream)
    other = context.target.get_forward_kv(context.layer_7, layer_gather, stream)
    torch_npu.npu.synchronize()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(other[0], layer_7_payload[0])
    assert torch.equal(other[1], layer_7_payload[1])
    return expected, layer_7_payload


def _assert_prepare_and_reuse(context: RuntimeContext, expected: TensorPair) -> None:
    gather = Batch(
        context.decode_mode,
        context.req,
        torch.tensor([6], device="npu"),
        context.active[:1],
        seq_lens=torch.tensor([7], device="npu"),
    )
    resident = context.target.prepare_resident(context.layer_8, gather, topk(1))
    torch_npu.npu.synchronize()
    assert resident.row_valid.tolist() == [True]
    assert bool((resident.current_slots[0, :7] >= 0).all().item())
    assert bool((resident.current_slots[0, 7:] == -1).all().item())
    slots = resident.current_slots[0, :7].to(torch.int64)
    assert torch.equal(resident.k[0, slots, 0], expected[0][:, 0])
    assert torch.equal(resident.rope[0, slots, 0], expected[1][:, 0])
    previous_identity = int(context.target.layers[1].last_req_ids[0].item())
    context.target_pool.reuse(1)
    replacement = (payload(7, 512, 131), payload(7, 64, 137))
    rewrite = Batch(
        context.extend_mode,
        context.req,
        torch.arange(7, device="npu"),
        context.active,
        extend_seq_lens=torch.tensor([7], device="npu"),
    )
    context.target.offload_v2(
        *replacement, context.layer_8, rewrite, torch.npu.current_stream()
    )
    resident = context.target.prepare_resident(context.layer_8, gather, topk(1))
    torch_npu.npu.synchronize()
    assert int(context.target.generations[1].item()) == int(
        context.target_pool.req_generation[1]
    )
    assert int(context.target.layers[1].last_req_ids[0].item()) != previous_identity
    slots = resident.current_slots[0, :7].to(torch.int64)
    assert torch.equal(resident.k[0, slots, 0], replacement[0][:, 0])
    assert not torch.equal(resident.k[0, slots, 0], expected[0][:, 0])


def _assert_mtp_rewrite(context: RuntimeContext) -> None:
    stream = torch.npu.current_stream()
    positions = torch.arange(7, 11, device="npu")
    mtp = Batch(
        context.target_verify_mode,
        context.req,
        positions,
        torch.tensor([1, 1, 1, 0], device="npu"),
    )
    initial = (payload(4, 512, 73), payload(4, 64, 79))
    context.target.offload_v2(*initial, context.layer_8, mtp, stream)
    first = context.target.prepare_resident(context.layer_8, mtp, topk(4))
    torch_npu.npu.synchronize()
    assert first.row_valid.tolist() == [True, True, True, False]
    assert bool((first.current_slots[3] == -1).all().item())
    rewrite = (payload(1, 512, 151), payload(1, 64, 157))
    rewrite_batch = Batch(
        context.decode_mode,
        context.req,
        torch.tensor([7], device="npu"),
        context.active[:1],
    )
    context.target.offload_v2(*rewrite, context.layer_8, rewrite_batch, stream)
    rewritten = context.target.prepare_resident(context.layer_8, mtp, topk(4))
    torch_npu.npu.synchronize()
    for row in range(3):
        token_7_slot = int(rewritten.current_slots[row, 7].item())
        assert torch.equal(rewritten.k[row, token_7_slot, 0], rewrite[0][0, 0])
    restored_value = (payload(1, 512, 181), payload(1, 64, 191))
    restore_batch = Batch(
        context.decode_mode,
        context.req,
        torch.tensor([10], device="npu"),
        context.active[:1],
    )
    context.target.offload_v2(*restored_value, context.layer_8, restore_batch, stream)
    restored_batch = Batch(
        context.target_verify_mode,
        context.req,
        positions,
        torch.ones(4, dtype=torch.int64, device="npu"),
    )
    restored = context.target.prepare_resident(context.layer_8, restored_batch, topk(4))
    torch_npu.npu.synchronize()
    assert restored.row_valid.tolist() == [True, True, True, True]
    token_10_slot = int(restored.current_slots[3, 10].item())
    assert torch.equal(restored.k[3, token_10_slot, 0], restored_value[0][0, 0])


def _assert_request_and_draft_isolation(
    context: RuntimeContext, layer_7_payload: TensorPair
) -> None:
    stream = torch.npu.current_stream()
    context.target_pool.reuse(2)
    req_2 = torch.tensor([2], dtype=torch.int64, device="npu")
    req_2_payload = (payload(2, 512, 211), payload(2, 64, 223))
    req_2_write = Batch(
        context.extend_mode,
        req_2,
        torch.arange(2, device="npu"),
        context.active[:2],
        extend_seq_lens=torch.tensor([2], device="npu"),
    )
    context.target.offload_v2(*req_2_payload, context.layer_7, req_2_write, stream)
    req_2_gather = Batch(
        context.decode_mode,
        req_2,
        torch.tensor([1], device="npu"),
        context.active[:1],
        seq_lens=torch.tensor([2], device="npu"),
    )
    req_1_gather = req_2_gather._replace(
        req_pool_indices=context.req, seq_lens=torch.tensor([5], device="npu")
    )
    req_2_actual = context.target.get_forward_kv(context.layer_7, req_2_gather, stream)
    req_1_actual = context.target.get_forward_kv(context.layer_7, req_1_gather, stream)
    torch_npu.npu.synchronize()
    assert torch.equal(req_2_actual[0], req_2_payload[0])
    assert torch.equal(req_2_actual[1], req_2_payload[1])
    assert torch.equal(req_1_actual[0], layer_7_payload[0])
    assert torch.equal(req_1_actual[1], layer_7_payload[1])

    draft = make_draft()
    assert draft.host_k_bases[0] != context.target.host_k_bases[0]
    draft_payload = (payload(2, 512, 233), payload(2, 64, 239))
    draft.offload_v2(*draft_payload, context.layer_7, req_2_write, stream)
    draft_actual = draft.get_forward_kv(context.layer_7, req_2_gather, stream)
    target_actual = context.target.get_forward_kv(context.layer_7, req_1_gather, stream)
    torch_npu.npu.synchronize()
    assert torch.equal(draft_actual[0], draft_payload[0])
    assert torch.equal(draft_actual[1], draft_payload[1])
    assert torch.equal(target_actual[0], layer_7_payload[0])
    assert torch.equal(target_actual[1], layer_7_payload[1])


def test_real_manager_multistep_lifecycle() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    context = make_context()
    assert_real_dva_and_factory_sharing(context)
    expected, layer_7_payload = _assert_prefill_and_decode(context)
    _assert_prepare_and_reuse(context, expected)
    _assert_mtp_rewrite(context)
    _assert_request_and_draft_isolation(context, layer_7_payload)
