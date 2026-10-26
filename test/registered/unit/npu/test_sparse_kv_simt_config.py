from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _EnvValue:
    def __init__(self, name: str, default: int | bool):
        self.name = name
        self.default = default

    def get(self) -> int | bool:
        raw = os.getenv(self.name)
        if raw is None:
            return self.default
        if isinstance(self.default, bool):
            return raw.lower() in {"1", "true", "yes", "y"}
        return int(raw)


def _load_config_module():
    module_name = "_sparse_kv_simt_config_under_test"
    sys.modules.pop(module_name, None)

    stubs = {
        "sglang.srt.configs.model_config": SimpleNamespace(
            get_dsa_index_head_dim=lambda config: config.index_head_dim,
            get_dsa_index_topk=lambda config: config.index_topk,
            is_deepseek_dsa=lambda _config: True,
        ),
        "sglang.srt.environ": SimpleNamespace(
            envs=SimpleNamespace(
                SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=_EnvValue(
                    "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD", False
                ),
                SGLANG_NPU_SPARSE_KV_SIMT=_EnvValue("SGLANG_NPU_SPARSE_KV_SIMT", False),
                SGLANG_NPU_SPARSE_KV_CAPACITY=_EnvValue(
                    "SGLANG_NPU_SPARSE_KV_CAPACITY", 4096
                ),
                SGLANG_NPU_USE_MLAPO=_EnvValue("SGLANG_NPU_USE_MLAPO", False),
            )
        ),
        "sglang.srt.runtime_context": SimpleNamespace(
            attention_backends=lambda: ("ascend", "ascend"),
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
            get_memory=lambda: SimpleNamespace(
                disable_radix_cache=True,
                enable_hierarchical_cache=False,
                enable_hisparse=False,
            ),
            get_parallel=lambda: SimpleNamespace(attn_dcp_size=1, attn_cp_size=1),
            get_schedule=lambda: SimpleNamespace(max_running_requests=32),
        ),
        "sglang.srt.hardware_backend.npu.utils": SimpleNamespace(
            is_npu_arch35=lambda: True
        ),
        "sglang.srt.utils.common": SimpleNamespace(is_npu=lambda: True),
    }
    with patch.dict(sys.modules, stubs):
        path = (
            Path(__file__).parents[4]
            / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/config.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def test_simt_capacity_defaults_to_4096_and_rejects_less_than_topk():
    config = _load_config_module()

    with patch.dict(os.environ, {"SGLANG_NPU_SPARSE_KV_SIMT": "1"}, clear=False):
        assert config.is_sparse_kv_simt_enabled()
        assert config.get_sparse_kv_simt_capacity(2048) == 4096

    with patch.dict(
        os.environ,
        {
            "SGLANG_NPU_SPARSE_KV_SIMT": "1",
            "SGLANG_NPU_SPARSE_KV_CAPACITY": "1024",
        },
        clear=False,
    ):
        with pytest.raises(ValueError, match="at least index_topk"):
            config.get_sparse_kv_simt_capacity(2048)

    with patch.dict(
        os.environ,
        {"SGLANG_NPU_SPARSE_KV_CAPACITY": "4100"},
        clear=False,
    ):
        with pytest.raises(ValueError, match="divisible by 128"):
            config.get_sparse_kv_simt_capacity(2048)


@pytest.mark.parametrize(("steps", "width"), [(None, 1), (0, 1), (3, 4)])
def test_simt_query_width_matches_supported_mtp_modes(steps, width):
    config = _load_config_module()
    server_args = SimpleNamespace(
        speculative_num_steps=steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )

    assert config.get_sparse_kv_simt_query_width(server_args) == width


def test_simt_query_width_rejects_unimplemented_mtp_shape():
    config = _load_config_module()

    with pytest.raises(ValueError, match="MTP=0 or MTP=3"):
        config.get_sparse_kv_simt_query_width(
            SimpleNamespace(
                speculative_num_steps=2,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=3,
            )
        )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"speculative_eagle_topk": 2}, "speculative_eagle_topk=1"),
        ({"speculative_num_draft_tokens": 3}, "num_draft_tokens=4"),
    ],
)
def test_simt_query_width_rejects_tree_or_verification_width(overrides, expected):
    config = _load_config_module()
    server_args = SimpleNamespace(
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    )
    for name, value in overrides.items():
        setattr(server_args, name, value)

    with pytest.raises(ValueError, match=expected):
        config.get_sparse_kv_simt_query_width(server_args)


def test_resident_budget_prices_bf16_main_planes_before_index_pool():
    config = _load_config_module()

    budget = config.get_sparse_kv_simt_resident_bytes(
        max_running_requests=32,
        query_width=4,
        capacity=4096,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_layers=78,
        topk=2048,
        max_context_len=140000,
    )
    max_rows = 33 * 4
    payload = max_rows * 4096 * (512 + 64) * 2 * 78
    request_tables = 33 * ((140000 + 127) // 128) * 4 * 2
    global_state = (
        4
        + max_rows * 2048 * 4 * 2
        + 33 * 8 * 2
        + max_rows * 4 * 3
        + request_tables
        + max_rows * (4096 // 128) * 4
        + max_rows * 4
        + 4
    )
    per_layer_metadata = (
        max_rows * 8
        + max_rows * 4096 * 4 * 2
        + max_rows * 4 * 2
        + max_rows * 2048 * 4 * 2
        + 16
    )

    assert budget > payload
    assert budget == payload + global_state + 78 * per_layer_metadata

    draft_budget = config.get_sparse_kv_simt_resident_bytes(
        max_running_requests=32,
        query_width=4,
        capacity=4096,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_layers=3,
        topk=2048,
        max_context_len=140000,
    )
    merged_budget = config.get_sparse_kv_simt_resident_bytes(
        max_running_requests=32,
        query_width=4,
        capacity=4096,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_layers=81,
        topk=2048,
        max_context_len=140000,
    )
    assert budget + draft_budget == merged_budget + global_state


def test_glm_freq4_c8_target_and_draft_index_pool_cell_size():
    config = _load_config_module()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_head_dim=128, index_topk=2048),
        index_head_dim=128,
    )

    with patch.dict(
        os.environ,
        {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
        clear=False,
    ):
        target_cell_size = config.get_sparsity_driven_kv_offload_cell_size(
            model_config=model_config,
            use_mla_backend=True,
            num_layers=21,
            element_size=2,
            index_element_size=1,
            index_scale_element_size=4,
        )

    index_budget = config.get_sparse_kv_index_pool_budget(
        target_num_layers=21,
        draft_num_layers=1,
        index_head_dim=128,
        index_element_size=1,
        index_scale_element_size=4,
        page_size=128,
    )

    assert target_cell_size == index_budget.target_cell_bytes == 21 * 132
    assert index_budget.draft_cell_bytes == 132
    assert index_budget.padding_bytes == 22 * 128 * 132


def test_simt_requires_sparse_kv_offload():
    config = _load_config_module()
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        attn_dcp_size=1,
        attn_cp_size=1,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        disable_radix_cache=True,
        speculative_num_steps=0,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
    )

    with (
        patch.dict(
            os.environ,
            {
                "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "0",
                "SGLANG_NPU_SPARSE_KV_SIMT": "1",
            },
            clear=False,
        ),
        pytest.raises(ValueError, match="ENABLE_SPARSE_KV_OFFLOAD"),
    ):
        config.validate_sparse_kv_simt_runtime(server_args)


def test_simt_rejects_mlapo_preprocessing_path():
    config = _load_config_module()
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        attn_dcp_size=1,
        attn_cp_size=1,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        disable_radix_cache=True,
        speculative_num_steps=0,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
    )

    with (
        patch.dict(
            os.environ,
            {
                "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1",
                "SGLANG_NPU_USE_MLAPO": "1",
            },
            clear=False,
        ),
        pytest.raises(ValueError, match="SGLANG_NPU_USE_MLAPO=0"),
    ):
        config.validate_sparse_kv_simt_runtime(server_args)


def test_simt_requires_arch35_and_bf16_main_cache():
    config = _load_config_module()
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        attn_dcp_size=1,
        attn_cp_size=1,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        disable_radix_cache=True,
        speculative_num_steps=0,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
    )

    with (
        patch.dict(
            os.environ,
            {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            clear=False,
        ),
        patch.object(config, "is_npu_arch35", return_value=False),
        pytest.raises(ValueError, match="architecture 35"),
    ):
        config.validate_sparse_kv_simt_runtime(server_args)

    with (
        patch.dict(
            os.environ,
            {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            clear=False,
        ),
        pytest.raises(ValueError, match="BF16 main KV cache"),
    ):
        config.validate_sparse_kv_simt_runtime(server_args, kv_cache_dtype="fp8_e4m3")

    with patch.dict(
        os.environ,
        {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
        clear=False,
    ):
        config.validate_sparse_kv_simt_runtime(server_args, kv_cache_dtype="bfloat16")


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"disaggregation_mode": "decode"}, "PD disaggregation"),
        ({"attn_dcp_size": 2}, "DCP"),
        ({"attn_cp_size": 2}, "context parallelism"),
        ({"enable_hierarchical_cache": True}, "HiCache"),
        ({"enable_hisparse": True}, "HiSparse"),
        ({"disable_radix_cache": False}, "prefix cache"),
    ],
)
def test_simt_guard_rejects_unimplemented_cache_combinations(overrides, expected):
    config = _load_config_module()
    server_args = SimpleNamespace(
        disaggregation_mode="null",
        attn_dcp_size=1,
        attn_cp_size=1,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        disable_radix_cache=True,
        speculative_num_steps=0,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
    )
    for name, value in overrides.items():
        setattr(server_args, name, value)

    with (
        patch.dict(
            os.environ,
            {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1"},
            clear=False,
        ),
        pytest.raises(ValueError, match=expected),
    ):
        config.validate_sparse_kv_simt_runtime(server_args)
