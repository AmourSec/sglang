from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def forward_sparse_kv_simt(
    backend: AscendAttnBackend,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    manager = backend.sparse_kv_manager
    resident = manager.prepare_resident(layer, forward_batch, topk_indices)
    rows = int(resident.current_slots.shape[0])
    num_query_heads = layer.tp_q_head_num
    padded_query_heads = q_nope.numel() // (rows * backend.kv_lora_rank)
    if padded_query_heads < num_query_heads:
        raise RuntimeError("Sparse KV query head storage is smaller than TP heads")
    query = q_nope.view(rows, padded_query_heads, backend.kv_lora_rank).contiguous()
    query_rope = q_rope.view(
        rows, padded_query_heads, backend.qk_rope_head_dim
    ).contiguous()
    sparse_indices = resident.current_slots.unsqueeze(1)
    empty_rows = ~resident.row_valid
    sparse_indices[:, :, 0].masked_fill_(empty_rows.view(rows, 1), 0)
    pages_per_row = manager.capacity // manager.block_size
    resident_k = resident.k.view(
        rows * pages_per_row,
        manager.block_size,
        1,
        backend.kv_lora_rank,
    )
    resident_rope = resident.rope.view(
        rows * pages_per_row,
        manager.block_size,
        1,
        backend.qk_rope_head_dim,
    )
    result = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=resident_k,
        value=resident_k,
        sparse_indices=sparse_indices,
        scale_value=layer.scaling,
        block_table=manager.resident_block_table[:rows],
        actual_seq_lengths_query=manager.resident_query_lens[:rows],
        actual_seq_lengths_kv=manager.visible_seq_lens[:rows],
        query_rope=query_rope,
        key_rope=resident_rope,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    output = result[0] if isinstance(result, tuple) else result
    output = output[:, :num_query_heads, :].reshape(
        rows, num_query_heads * backend.kv_lora_rank
    )
    return torch.where(
        resident.row_valid.view(rows, 1), output, torch.zeros_like(output)
    )
