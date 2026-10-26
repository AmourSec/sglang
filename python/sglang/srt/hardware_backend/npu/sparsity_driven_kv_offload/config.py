"""Configuration and validation for sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

from sglang.srt.configs.model_config import (
    get_dsa_index_head_dim,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.runtime_context import attention_backends, get_schedule
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


class SparseKVIndexPoolBudget(NamedTuple):
    target_cell_bytes: int
    draft_cell_bytes: int
    padding_bytes: int


def get_sparse_kv_index_pool_budget(
    *,
    target_num_layers: int,
    draft_num_layers: int,
    index_head_dim: int,
    index_element_size: int,
    index_scale_element_size: int,
    page_size: int,
) -> SparseKVIndexPoolBudget:
    bytes_per_layer_token = (
        index_head_dim * index_element_size + index_scale_element_size
    )
    return SparseKVIndexPoolBudget(
        target_cell_bytes=target_num_layers * bytes_per_layer_token,
        draft_cell_bytes=draft_num_layers * bytes_per_layer_token,
        padding_bytes=(target_num_layers + draft_num_layers)
        * page_size
        * bytes_per_layer_token,
    )


def is_sparse_kv_simt_enabled() -> bool:
    return bool(envs.SGLANG_NPU_SPARSE_KV_SIMT.get())


def get_sparse_kv_simt_capacity(index_topk: int) -> int:
    capacity = int(envs.SGLANG_NPU_SPARSE_KV_CAPACITY.get())
    if capacity < index_topk:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_CAPACITY must be at least index_topk, "
            f"got capacity={capacity}, index_topk={index_topk}."
        )
    if capacity % 128:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_CAPACITY must be divisible by 128, "
            f"got capacity={capacity}."
        )
    return capacity


def get_sparse_kv_simt_query_width(server_args: ServerArgs) -> int:
    steps = server_args.speculative_num_steps
    if steps is None or int(steps) == 0:
        return 1
    if int(steps) == 3:
        if server_args.speculative_eagle_topk != 1:
            raise ValueError(
                "SGLANG_NPU_SPARSE_KV_SIMT requires speculative_eagle_topk=1 for MTP=3."
            )
        if server_args.speculative_num_draft_tokens != 4:
            raise ValueError(
                "SGLANG_NPU_SPARSE_KV_SIMT requires "
                "speculative_num_draft_tokens=4 for MTP=3."
            )
        return 4
    raise ValueError(
        "SGLANG_NPU_SPARSE_KV_SIMT currently supports only MTP=0 or MTP=3, "
        f"got speculative_num_steps={steps}."
    )


def get_sparse_kv_simt_resident_bytes(
    *,
    max_running_requests: int,
    query_width: int,
    capacity: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_layers: int,
    topk: int,
    max_context_len: int,
    block_size: int = 128,
) -> int:
    request_slots = max_running_requests + 1
    max_rows = request_slots * query_width
    pages_per_request = (max_context_len + block_size - 1) // block_size
    resident_pages_per_row, remainder = divmod(capacity, block_size)
    if remainder:
        raise ValueError("Sparse KV capacity must be divisible by block_size.")

    hash_capacity = 32
    while hash_capacity < 2 * topk:
        hash_capacity <<= 1
    workspace_elements_per_row = 3 * hash_capacity + 2 * capacity + topk
    workspace_elements = (
        1
        if workspace_elements_per_row * 4 <= 112 * 1024
        else max_rows * workspace_elements_per_row
    )

    global_bytes = (
        4
        + max_rows * topk * 4 * 2
        + request_slots * 8 * 2
        + max_rows * 4 * 3
        + request_slots * pages_per_request * 4 * 2
        + max_rows * resident_pages_per_row * 4
        + max_rows * 4
        + workspace_elements * 4
    )
    per_layer_bytes = (
        max_rows * capacity * (kv_lora_rank + qk_rope_head_dim) * 2
        + max_rows * 8
        + max_rows * capacity * 4 * 2
        + max_rows * 4 * 2
        + max_rows * topk * 4 * 2
        + 2 * 8
    )
    return global_bytes + num_layers * per_layer_bytes


def validate_sparse_kv_simt_runtime(
    server_args: ServerArgs, *, kv_cache_dtype: object | None = None
) -> None:
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_SIMT requires SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1."
        )
    if envs.SGLANG_NPU_USE_MLAPO.get():
        raise ValueError("SGLANG_NPU_SPARSE_KV_SIMT requires SGLANG_NPU_USE_MLAPO=0.")
    if not is_npu_arch35():
        raise ValueError("SGLANG_NPU_SPARSE_KV_SIMT requires NPU architecture 35.")
    if kv_cache_dtype is not None and str(kv_cache_dtype).lower() not in {
        "bf16",
        "bfloat16",
        "torch.bfloat16",
    }:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_SIMT requires a BF16 main KV cache, "
            f"got {kv_cache_dtype}."
        )
    if server_args.disaggregation_mode != "null":
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_SIMT does not support PD disaggregation."
        )
    if server_args.attn_dcp_size != 1:
        raise ValueError("SGLANG_NPU_SPARSE_KV_SIMT does not support DCP.")
    if server_args.attn_cp_size != 1:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_SIMT does not support context parallelism."
        )
    if server_args.enable_hierarchical_cache:
        raise ValueError("SGLANG_NPU_SPARSE_KV_SIMT does not support HiCache.")
    if server_args.enable_hisparse:
        raise ValueError("SGLANG_NPU_SPARSE_KV_SIMT does not support HiSparse.")
    if not server_args.disable_radix_cache:
        raise ValueError(
            "SGLANG_NPU_SPARSE_KV_SIMT requires the prefix cache to be disabled."
        )
    get_sparse_kv_simt_query_width(server_args)


def is_sparsity_driven_kv_offload_enabled(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
) -> bool:
    if not envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get():
        return False

    prefill_attention_backend, decode_attention_backend = attention_backends()
    if not (
        is_npu()
        and prefill_attention_backend == "ascend"
        and decode_attention_backend == "ascend"
        and use_mla_backend
        and is_deepseek_dsa(model_config.hf_config)
    ):
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires an NPU "
            "DSA-family MLA model "
            "(for example DeepSeek V3.2 or GLM-5.x) using the Ascend MLA "
            "attention backend."
        )
    if get_schedule().max_running_requests is None:
        raise ValueError(
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD requires max_running_requests "
            "to be set to bound the per-process host KV allocation."
        )
    return True


def get_sparsity_driven_kv_offload_sparse_context_len(
    *,
    model_config: ModelConfig,
) -> int:
    """Return the per-request on-device sparse KV window size."""
    sparse_context_len = int(get_dsa_index_topk(model_config.hf_config))
    if sparse_context_len <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_topk, "
            f"got {sparse_context_len}."
        )
    return sparse_context_len


def get_sparsity_driven_kv_offload_index_head_dim(
    *,
    model_config: ModelConfig,
) -> int:
    index_head_dim = getattr(model_config, "index_head_dim", None)
    if index_head_dim is None:
        index_head_dim = get_dsa_index_head_dim(model_config.hf_config)
    index_head_dim = int(index_head_dim)
    if index_head_dim <= 0:
        raise ValueError(
            "Sparsity-driven KV offload requires a positive DSA index_head_dim, "
            f"got {index_head_dim}."
        )
    return index_head_dim


def get_sparsity_driven_kv_offload_cell_size(
    *,
    model_config: ModelConfig,
    use_mla_backend: bool,
    num_layers: int,
    element_size: int,
    index_element_size: int | None = None,
    index_scale_element_size: int = 0,
) -> Optional[int]:
    if not is_sparsity_driven_kv_offload_enabled(
        model_config=model_config,
        use_mla_backend=use_mla_backend,
    ):
        return None

    index_head_dim = get_sparsity_driven_kv_offload_index_head_dim(
        model_config=model_config
    )
    resolved_index_element_size = (
        element_size if index_element_size is None else index_element_size
    )
    return get_sparse_kv_index_pool_budget(
        target_num_layers=num_layers,
        draft_num_layers=0,
        index_head_dim=index_head_dim,
        index_element_size=resolved_index_element_size,
        index_scale_element_size=index_scale_element_size,
        page_size=0,
    ).target_cell_bytes
