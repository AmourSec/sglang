from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple, Protocol

import torch

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt import (
        SparseKVSimtManager,
    )

TOPK = 2048
CAPACITY = 2048
MAX_CONTEXT = 32


class ForwardModeLike(Protocol):
    def is_extend_without_speculative(self) -> bool: ...


class ReqPool:
    def __init__(self) -> None:
        self.req_to_token = torch.zeros(
            (4, MAX_CONTEXT), dtype=torch.int32, device="npu"
        )
        self.max_context_len = MAX_CONTEXT
        self.device = torch.device("npu")
        self.req_generation = torch.zeros(4, dtype=torch.int64)
        self.req_generation[1] = 1
        self._on_alloc_rows: Callable[[list[int]], None] | None = None

    def reuse(self, index: int) -> None:
        self.req_generation[index] += 1
        if self._on_alloc_rows is not None:
            self._on_alloc_rows([index])


class PagedPool(NamedTuple):
    start_layer: int
    layer_num: int
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    store_dtype: torch.dtype = torch.bfloat16


class Layer(NamedTuple):
    layer_id: int


class Batch(NamedTuple):
    forward_mode: ForwardModeLike
    req_pool_indices: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor | None = None
    extend_seq_lens: torch.Tensor | None = None


class Allocator(NamedTuple):
    pool: PagedPool

    def get_kvcache(self) -> PagedPool:
        return self.pool


class Runner(NamedTuple):
    token_to_kv_pool_allocator: Allocator


class RuntimeContext(NamedTuple):
    target: SparseKVSimtManager
    target_pool: ReqPool
    target_paged: PagedPool
    layer_7: Layer
    layer_8: Layer
    req: torch.Tensor
    active: torch.Tensor
    extend_mode: ForwardModeLike
    decode_mode: ForwardModeLike
    target_verify_mode: ForwardModeLike


def payload(rows: int, dim: int, seed: int) -> torch.Tensor:
    values = torch.arange(rows * dim, dtype=torch.float32).reshape(rows, 1, dim)
    return ((values.remainder(251) + seed) / 257).to(torch.bfloat16).to("npu")


def topk(rows: int) -> torch.Tensor:
    return torch.arange(TOPK, dtype=torch.int32, device="npu").repeat(rows, 1)


def make_context() -> RuntimeContext:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt import (
        SparseKVSimtManager,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    target_pool = ReqPool()
    target_paged = PagedPool(start_layer=7, layer_num=2)
    target = SparseKVSimtManager(
        req_to_token_pool=target_pool,
        paged_kv_cache=target_paged,
        capacity=CAPACITY,
        topk=TOPK,
        query_width=4,
        host_reserve_layer_num=3,
    )
    return RuntimeContext(
        target=target,
        target_pool=target_pool,
        target_paged=target_paged,
        layer_7=Layer(7),
        layer_8=Layer(8),
        req=torch.tensor([1], dtype=torch.int64, device="npu"),
        active=torch.ones(7, dtype=torch.int64, device="npu"),
        extend_mode=ForwardMode.EXTEND,
        decode_mode=ForwardMode.DECODE,
        target_verify_mode=ForwardMode.TARGET_VERIFY,
    )


def assert_real_dva_and_factory_sharing(context: RuntimeContext) -> None:
    from memfabric_hybrid import offload

    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
        _pool_sparse_kv_managers,
        get_or_create_sparse_kv_manager,
    )

    target = context.target
    assert target.host_k[0].device.type == "cpu"
    assert target.host_rope[0].device.type == "cpu"
    assert target.host_k_bases[0] == offload.get_dva(target.host_k[0].data_ptr())
    assert target.host_rope_bases[0] == offload.get_dva(target.host_rope[0].data_ptr())
    assert target.host_k_bases[0] != 0 and target.host_rope_bases[0] != 0
    pool_key = id(context.target_paged)
    _pool_sparse_kv_managers[pool_key] = target
    try:
        runner = Runner(Allocator(context.target_paged))
        assert get_or_create_sparse_kv_manager(runner) is target
        assert get_or_create_sparse_kv_manager(runner) is target
    finally:
        _pool_sparse_kv_managers.pop(pool_key, None)


def make_draft() -> SparseKVSimtManager:
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.simt import (
        SparseKVSimtManager,
    )

    return SparseKVSimtManager(
        req_to_token_pool=ReqPool(),
        paged_kv_cache=PagedPool(start_layer=7, layer_num=1),
        capacity=CAPACITY,
        topk=TOPK,
        query_width=1,
        host_reserve_layer_num=1,
    )
