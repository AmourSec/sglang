from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt import (
        SparseKVSimtManager,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def offload_rows(
    manager: SparseKVSimtManager,
    k: torch.Tensor,
    k_rope: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    stream: torch.npu.Stream,
) -> None:
    rows = int(k.shape[0])
    query_lens = manager._query_lens(forward_batch, rows)
    token_to_req = torch.repeat_interleave(
        torch.arange(query_lens.numel(), device=manager.device),
        query_lens,
        output_size=rows,
    )
    request_slots = forward_batch.req_pool_indices.to(torch.int64)
    positions = forward_batch.positions.to(torch.int64)
    row_slots = request_slots.index_select(0, token_to_req)
    destination = row_slots * manager.host_stride + positions
    source = torch.arange(rows, dtype=torch.int64, device=manager.device)
    valid = (
        (row_slots > 0)
        & (row_slots < manager.size)
        & (positions >= 0)
        & (positions < manager.max_context_len)
        & (forward_batch.out_cache_loc[:rows] > 0)
    )
    layer_index = layer.layer_id - manager.start_layer
    with torch.npu.stream(stream):
        unidex_copy_inplace(
            k.contiguous(),
            manager.host_k[layer_index],
            source,
            destination,
            valid,
            1,
            2,
            block_dim=48,
            dst_ptr=manager.host_k_bases[layer_index],
        )
        unidex_copy_inplace(
            k_rope.contiguous(),
            manager.host_rope[layer_index],
            source,
            destination,
            valid,
            1,
            2,
            block_dim=48,
            dst_ptr=manager.host_rope_bases[layer_index],
        )


def gather_forward_kv(
    manager: SparseKVSimtManager,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    stream: torch.npu.Stream,
) -> tuple[torch.Tensor, torch.Tensor]:
    request_slots = forward_batch.req_pool_indices.to(torch.int64)
    seq_lens = forward_batch.seq_lens.to(torch.int64)
    total = int(seq_lens.sum().item())
    requests = torch.repeat_interleave(request_slots, seq_lens)
    starts = torch.repeat_interleave(torch.cumsum(seq_lens, 0) - seq_lens, seq_lens)
    positions = torch.arange(total, device=manager.device) - starts
    source = requests * manager.host_stride + positions
    destination = torch.arange(total, dtype=torch.int64, device=manager.device)
    valid = torch.ones(total, dtype=torch.bool, device=manager.device)
    k = torch.empty(
        (total, 1, manager.kv_lora_rank),
        dtype=torch.bfloat16,
        device=manager.device,
    )
    rope = torch.empty(
        (total, 1, manager.qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=manager.device,
    )
    layer_index = layer.layer_id - manager.start_layer
    with torch.npu.stream(stream):
        unidex_copy_inplace(
            manager.host_k[layer_index],
            k,
            source,
            destination,
            valid,
            2,
            1,
            block_dim=48,
            src_ptr=manager.host_k_bases[layer_index],
        )
        unidex_copy_inplace(
            manager.host_rope[layer_index],
            rope,
            source,
            destination,
            valid,
            2,
            1,
            block_dim=48,
            src_ptr=manager.host_rope_bases[layer_index],
        )
    return k, rope
