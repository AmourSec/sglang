from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.srt.mem_cache.pool_host.npu_memfabric import (
    alloc_with_memfabric,
    ensure_memfabric_capacity,
)


class SparseKVLayerState(NamedTuple):
    resident_k: torch.Tensor
    resident_rope: torch.Tensor
    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    host_cache_bases: torch.Tensor
    stable_frontiers: torch.Tensor


def _workspace_elements(topk: int, capacity: int) -> int:
    hash_capacity = 32
    while hash_capacity < 2 * topk:
        hash_capacity <<= 1
    return 3 * hash_capacity + 2 * capacity + topk


class SparseKVSimtState:
    def __init__(
        self,
        *,
        size: int,
        max_context_len: int,
        max_rows: int,
        capacity: int,
        topk: int,
        layer_num: int,
        host_reserve_layer_num: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        device: torch.device,
    ) -> None:
        from memfabric_hybrid import offload

        self.block_size = 128
        self.pages_per_request = (
            max_context_len + self.block_size - 1
        ) // self.block_size
        self.host_num_blocks = size * self.pages_per_request
        self.host_stride = self.pages_per_request * self.block_size
        k_shape = (self.host_num_blocks, self.block_size, 1, kv_lora_rank)
        rope_shape = (
            self.host_num_blocks,
            self.block_size,
            1,
            qk_rope_head_dim,
        )
        per_layer_bytes = (
            torch.Size(k_shape).numel() + torch.Size(rope_shape).numel()
        ) * torch.bfloat16.itemsize
        if host_reserve_layer_num < layer_num:
            raise RuntimeError("Sparse KV host reservation is smaller than its layers")
        ensure_memfabric_capacity(
            per_layer_bytes * host_reserve_layer_num, torch.npu.current_device()
        )
        self.host_k = []
        self.host_rope = []
        self.host_k_bases = []
        self.host_rope_bases = []
        for _ in range(layer_num):
            host_k = alloc_with_memfabric(k_shape, torch.bfloat16, "cpu", True, None)
            host_rope = alloc_with_memfabric(
                rope_shape, torch.bfloat16, "cpu", True, None
            )
            k_dva = int(offload.get_dva(host_k.data_ptr()))
            rope_dva = int(offload.get_dva(host_rope.data_ptr()))
            if k_dva == 0 or rope_dva == 0:
                raise RuntimeError("MemFabric failed to map sparse KV host planes")
            self.host_k.append(host_k)
            self.host_rope.append(host_rope)
            self.host_k_bases.append(k_dva)
            self.host_rope_bases.append(rope_dva)

        self.active_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.current_slots = torch.full(
            (max_rows, topk), -1, dtype=torch.int32, device=device
        )
        self.req_ids = torch.zeros(size, dtype=torch.int64, device=device)
        self.generations = torch.zeros(size, dtype=torch.int64, device=device)
        self.plan_topk = torch.full(
            (max_rows, topk), -1, dtype=torch.int32, device=device
        )
        self.stable_prefix_lens = torch.zeros(
            max_rows, dtype=torch.int32, device=device
        )
        self.visible_seq_lens = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.token_to_req = torch.zeros(max_rows, dtype=torch.int32, device=device)
        page_offsets = torch.arange(
            self.pages_per_request, dtype=torch.int32, device=device
        )
        request_offsets = torch.arange(size, dtype=torch.int32, device=device).view(
            -1, 1
        )
        self.block_table = (
            request_offsets * self.pages_per_request + page_offsets.view(1, -1)
        ).contiguous()
        self.plan_block_table = torch.empty_like(self.block_table)
        resident_pages_per_row = capacity // self.block_size
        if resident_pages_per_row * self.block_size != capacity:
            raise RuntimeError("Sparse KV capacity must be divisible by block size")
        self.resident_block_table = torch.arange(
            max_rows * resident_pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(max_rows, resident_pages_per_row)
        self.resident_query_lens = torch.arange(
            1, max_rows + 1, dtype=torch.int32, device=device
        )
        row_elements = _workspace_elements(topk, capacity)
        workspace_elements = (
            1
            if row_elements * torch.int32.itemsize <= 112 * 1024
            else max_rows * row_elements
        )
        self.workspace = torch.empty(
            workspace_elements, dtype=torch.int32, device=device
        )
        initial_lru = torch.arange(capacity, dtype=torch.int32, device=device)
        self.layers = [
            SparseKVLayerState(
                resident_k=torch.empty(
                    (max_rows, capacity, 1, kv_lora_rank),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                resident_rope=torch.empty(
                    (max_rows, capacity, 1, qk_rope_head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                last_req_ids=torch.full(
                    (max_rows,), -1, dtype=torch.int64, device=device
                ),
                slot_to_token=torch.full(
                    (max_rows, capacity), -1, dtype=torch.int32, device=device
                ),
                lru_slots=initial_lru.view(1, -1).repeat(max_rows, 1),
                miss_count=torch.zeros(max_rows, dtype=torch.int32, device=device),
                miss_tokens=torch.full(
                    (max_rows, topk), -1, dtype=torch.int32, device=device
                ),
                miss_slots=torch.full(
                    (max_rows, topk), -1, dtype=torch.int32, device=device
                ),
                host_cache_bases=torch.tensor(
                    [self.host_k_bases[layer], self.host_rope_bases[layer]],
                    dtype=torch.int64,
                    device=device,
                ),
                stable_frontiers=torch.zeros(
                    max_rows, dtype=torch.int32, device=device
                ),
            )
            for layer in range(layer_num)
        ]
