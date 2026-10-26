from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.metadata import (
    SparseKVMetadataInput,
    advance_stable_frontiers,
    build_request_identities,
    build_sparse_kv_metadata,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_state import (
    SparseKVSimtState,
)

if TYPE_CHECKING:
    import torch.npu

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_generation_listeners: dict[int, list[torch.Tensor]] = {}
_generation_hooked_pools: set[int] = set()


class SparseKVResident(NamedTuple):
    k: torch.Tensor
    rope: torch.Tensor
    current_slots: torch.Tensor
    row_valid: torch.Tensor


class SparseKVSimtManager:
    uses_simt = True

    def __init__(
        self,
        *,
        req_to_token_pool,
        paged_kv_cache,
        capacity: int,
        topk: int,
        query_width: int,
        host_reserve_layer_num: int,
    ) -> None:
        from sgl_kernel_npu_sparse_kv import ensure_loaded

        ensure_loaded()
        self.size = int(req_to_token_pool.req_to_token.shape[0])
        self.max_context_len = int(req_to_token_pool.max_context_len)
        self.capacity = capacity
        self.sparse_context_len = capacity
        self.topk = topk
        self.query_width = query_width
        self.max_rows = self.size * query_width
        self.device = req_to_token_pool.device
        self.paged_kv_cache = paged_kv_cache
        self.start_layer = paged_kv_cache.start_layer
        self.layer_num = paged_kv_cache.layer_num
        self.kv_lora_rank = paged_kv_cache.kv_lora_rank
        self.qk_rope_head_dim = paged_kv_cache.qk_rope_head_dim
        self.store_dtype = torch.bfloat16
        state = SparseKVSimtState(
            size=self.size,
            max_context_len=self.max_context_len,
            max_rows=self.max_rows,
            capacity=capacity,
            topk=topk,
            layer_num=self.layer_num,
            host_reserve_layer_num=host_reserve_layer_num,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            device=self.device,
        )
        self.block_size = state.block_size
        self.pages_per_request = state.pages_per_request
        self.host_num_blocks = state.host_num_blocks
        self.host_stride = state.host_stride
        self.host_k = state.host_k
        self.host_rope = state.host_rope
        self.host_k_bases = state.host_k_bases
        self.host_rope_bases = state.host_rope_bases
        self.active_rows = state.active_rows
        self.current_slots = state.current_slots
        self.req_ids = state.req_ids
        self.generations = state.generations
        self.plan_topk = state.plan_topk
        self.stable_prefix_lens = state.stable_prefix_lens
        self.visible_seq_lens = state.visible_seq_lens
        self.token_to_req = state.token_to_req
        self.block_table = state.block_table
        self.plan_block_table = state.plan_block_table
        self.resident_block_table = state.resident_block_table
        self.resident_query_lens = state.resident_query_lens
        self.workspace = state.workspace
        self.layers = state.layers
        self.req_to_token_pool = req_to_token_pool
        self.generations.copy_(req_to_token_pool.req_generation.to(self.device))
        self._install_alloc_hook(req_to_token_pool)

    def _install_alloc_hook(self, req_to_token_pool) -> None:
        pool_key = id(req_to_token_pool)
        listeners = _generation_listeners.setdefault(pool_key, [])
        if pool_key not in _generation_hooked_pools:
            original_hook = req_to_token_pool._on_alloc_rows

            def notify_generation(indices: list[int]) -> None:
                if original_hook is not None:
                    original_hook(indices)
                cpu_values = req_to_token_pool.req_generation[indices]
                for target in listeners:
                    device_indices = torch.tensor(
                        indices, dtype=torch.int64, device=target.device
                    )
                    target.index_copy_(0, device_indices, cpu_values.to(target.device))

            req_to_token_pool._on_alloc_rows = notify_generation
            _generation_hooked_pools.add(pool_key)
        listeners.append(self.generations)

    def _query_lens(self, forward_batch: ForwardBatch, rows: int) -> torch.Tensor:
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.extend_seq_lens is not None
        ):
            query_lens = forward_batch.extend_seq_lens.to(torch.int64)
            if int(query_lens.sum().item()) == rows:
                return query_lens
        requests = int(forward_batch.req_pool_indices.numel())
        if requests == 0 or rows % requests != 0:
            raise RuntimeError("Sparse KV rows do not form a linear request layout")
        return torch.full(
            (requests,), rows // requests, dtype=torch.int64, device=self.device
        )

    def offload_v2(
        self,
        k: torch.Tensor,
        k_rope: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ) -> None:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_transfer import (
            offload_rows,
        )

        offload_rows(self, k, k_rope, layer, forward_batch, stream)

    def prepare_resident(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
    ) -> SparseKVResident:
        topk = topk_indices.reshape(-1, topk_indices.shape[-1]).to(torch.int32)
        rows = int(topk.shape[0])
        if rows > self.max_rows or int(topk.shape[1]) != self.topk:
            raise RuntimeError("Sparse KV top-k shape exceeds persistent buffers")
        query_lens = self._query_lens(forward_batch, rows)
        request_slots = forward_batch.req_pool_indices.to(torch.int64)
        request_valid = request_slots.gt(0) & request_slots.lt(self.size)
        safe_request_slots = request_slots.clamp(0, self.size - 1)
        identities = build_request_identities(
            self.generations, safe_request_slots, request_valid
        )
        positions = forward_batch.positions[:rows].to(torch.int32)
        metadata = build_sparse_kv_metadata(
            SparseKVMetadataInput(
                request_slots=safe_request_slots,
                request_identities=identities,
                request_valid=request_valid,
                row_execution_valid=forward_batch.out_cache_loc[:rows].gt(0),
                query_lens=query_lens,
                positions=positions,
                topk_indices=topk,
                stable_prefix_lens=self.stable_prefix_lens[:rows],
                max_context_len=self.max_context_len,
                block_size=self.block_size,
            )
        )
        row_active = metadata.row_valid
        self.req_ids[: identities.numel()].copy_(metadata.req_ids)
        self.plan_topk[:rows].copy_(metadata.topk_indices)
        self.visible_seq_lens[:rows].copy_(metadata.visible_seq_lens)
        self.token_to_req[:rows].copy_(metadata.token_to_req)
        request_count = int(identities.numel())
        self.plan_block_table[:request_count].copy_(metadata.block_table)
        self.active_rows.fill_(rows)
        state = self.layers[layer.layer_id - self.start_layer]
        self.stable_prefix_lens[:rows].copy_(
            advance_stable_frontiers(
                state.stable_frontiers,
                state.last_req_ids,
                identities,
                request_valid,
                metadata.token_to_req,
                positions,
                row_active,
                self.max_context_len,
            )
        )
        torch.ops.sgl_sparse_kv.plan_transfer(
            self.req_ids[: identities.numel()],
            self.plan_topk[:rows],
            self.stable_prefix_lens[:rows],
            self.visible_seq_lens[:rows],
            self.token_to_req[:rows],
            self.plan_block_table[:request_count],
            self.active_rows,
            state.last_req_ids[:rows],
            state.slot_to_token[:rows],
            state.lru_slots[:rows],
            self.current_slots[:rows],
            state.miss_count[:rows],
            state.miss_tokens[:rows],
            state.miss_slots[:rows],
            self.workspace,
            state.host_cache_bases,
            state.resident_k[:rows],
            state.resident_rope[:rows],
            self.topk,
            self.capacity,
            self.max_context_len,
            self.block_size,
            self.host_num_blocks,
            self.kv_lora_rank * torch.bfloat16.itemsize,
            self.qk_rope_head_dim * torch.bfloat16.itemsize,
        )
        state.stable_frontiers[:rows].copy_(self.visible_seq_lens[:rows])
        row_valid = row_active & self.current_slots[:rows].ge(0).any(dim=1)
        return SparseKVResident(
            state.resident_k[:rows],
            state.resident_rope[:rows],
            self.current_slots[:rows],
            row_valid,
        )

    def get_forward_kv(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stream: torch.npu.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt_transfer import (
            gather_forward_kv,
        )

        return gather_forward_kv(self, layer, forward_batch, stream)
