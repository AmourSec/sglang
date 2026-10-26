from __future__ import annotations

from typing import NamedTuple, Protocol

import torch


class SparseKVMetadata(NamedTuple):
    req_ids: torch.Tensor
    topk_indices: torch.Tensor
    stable_prefix_lens: torch.Tensor
    visible_seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    block_table: torch.Tensor
    row_valid: torch.Tensor


class SparseKVMetadataInput(NamedTuple):
    request_slots: torch.Tensor
    request_identities: torch.Tensor
    request_valid: torch.Tensor
    row_execution_valid: torch.Tensor
    query_lens: torch.Tensor
    positions: torch.Tensor
    topk_indices: torch.Tensor
    stable_prefix_lens: torch.Tensor
    max_context_len: int
    block_size: int


class SparseKVForwardMode(Protocol):
    def is_decode_or_idle(self) -> bool: ...

    def is_target_verify(self) -> bool: ...

    def is_draft_extend_v2(self) -> bool: ...


def build_request_identities(
    generations: torch.Tensor,
    request_slots: torch.Tensor,
    request_valid: torch.Tensor,
) -> torch.Tensor:
    safe_slots = request_slots.clamp(0, generations.numel() - 1)
    request_generations = generations.index_select(0, safe_slots)
    identities = torch.bitwise_or(
        torch.bitwise_left_shift(request_generations, 32), safe_slots
    )
    return torch.where(request_valid, identities, torch.zeros_like(identities))


def pad_topk_rows(topk_indices: torch.Tensor, num_rows: int) -> torch.Tensor:
    current_rows = int(topk_indices.shape[0])
    if current_rows == num_rows:
        return topk_indices
    if current_rows > num_rows:
        raise RuntimeError("Sparse KV top-k rows exceed the padded query rows")
    padding = torch.full(
        (num_rows - current_rows, topk_indices.shape[1]),
        -1,
        dtype=topk_indices.dtype,
        device=topk_indices.device,
    )
    return torch.cat((topk_indices, padding), dim=0)


def should_write_host_kv(save_kv_cache: bool, uses_simt: bool) -> bool:
    return save_kv_cache or uses_simt


def should_use_sparse_kv_simt(forward_mode: SparseKVForwardMode) -> bool:
    return (
        forward_mode.is_decode_or_idle()
        or forward_mode.is_target_verify()
        or forward_mode.is_draft_extend_v2()
    )


def advance_stable_frontiers(
    stable_frontiers: torch.Tensor,
    last_req_ids: torch.Tensor,
    request_identities: torch.Tensor,
    request_valid: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    row_valid: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    sentinel = torch.full_like(positions, max_context_len)
    write_frontiers = torch.where(row_valid, positions, sentinel)
    request_frontiers = torch.full(
        request_identities.shape,
        max_context_len,
        dtype=torch.int32,
        device=positions.device,
    )
    request_frontiers.scatter_reduce_(
        0, token_to_req.to(torch.int64), write_frontiers, reduce="amin"
    )
    identity_matches = last_req_ids.view(-1, 1).eq(request_identities.view(1, -1))
    identity_matches &= request_valid.view(1, -1)
    pending_frontiers = torch.where(
        identity_matches,
        request_frontiers.view(1, -1),
        max_context_len,
    ).amin(dim=1)
    stable_frontiers.copy_(torch.minimum(stable_frontiers, pending_frontiers))
    return torch.minimum(stable_frontiers[: positions.numel()], positions)


def build_sparse_kv_metadata(inputs: SparseKVMetadataInput) -> SparseKVMetadata:
    device = inputs.positions.device
    token_to_req = torch.repeat_interleave(
        torch.arange(inputs.request_slots.numel(), dtype=torch.int32, device=device),
        inputs.query_lens.to(device=device, dtype=torch.int64),
        output_size=inputs.positions.numel(),
    )
    num_rows = int(token_to_req.numel())
    if inputs.positions.numel() != num_rows:
        raise RuntimeError("Sparse KV positions do not match query_lens")
    if inputs.topk_indices.shape[0] != num_rows:
        raise RuntimeError("Sparse KV top-k rows do not match query_lens")
    if inputs.stable_prefix_lens.numel() != num_rows:
        raise RuntimeError("Sparse KV stable frontiers do not match query_lens")

    request_rows = token_to_req.to(torch.int64)
    row_valid = inputs.request_valid.to(device=device, dtype=torch.bool).index_select(
        0, request_rows
    ) & inputs.row_execution_valid.to(device=device, dtype=torch.bool)
    req_ids = inputs.request_identities.to(device=device, dtype=torch.int64)
    visible_seq_lens = torch.where(
        row_valid,
        inputs.positions.to(device=device, dtype=torch.int32) + 1,
        torch.zeros(num_rows, dtype=torch.int32, device=device),
    )
    topk_indices = inputs.topk_indices.to(device=device, dtype=torch.int32).clone()
    topk_indices[~row_valid] = -1

    pages_per_request = (
        inputs.max_context_len + inputs.block_size - 1
    ) // inputs.block_size
    page_offsets = torch.arange(pages_per_request, dtype=torch.int32, device=device)
    block_table = inputs.request_slots.to(device=device, dtype=torch.int32).view(
        -1, 1
    ) * pages_per_request + page_offsets.view(1, -1)
    return SparseKVMetadata(
        req_ids=req_ids.contiguous(),
        topk_indices=topk_indices.contiguous(),
        stable_prefix_lens=inputs.stable_prefix_lens.to(
            device=device, dtype=torch.int32
        ).contiguous(),
        visible_seq_lens=visible_seq_lens.contiguous(),
        token_to_req=token_to_req.contiguous(),
        block_table=block_table.contiguous(),
        row_valid=row_valid.contiguous(),
    )
