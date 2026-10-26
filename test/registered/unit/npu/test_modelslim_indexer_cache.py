from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_indexer_cache_module():
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/layers/quantization/modelslim/indexer_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_modelslim_indexer_cache_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_dynamic_indexer_quantization_applies_to_all_layers():
    cache = _load_indexer_cache_module()

    assert cache.get_modelslim_c8_indexer_layers({}, (0, 3, 6)) == (0, 3, 6)


def test_layer_bf16_override_is_not_misclassified_as_c8():
    cache = _load_indexer_cache_module()

    assert (
        cache.get_modelslim_c8_indexer_layers(
            {
                "indexer_quant_type": "INT8_DYNAMIC",
                "model.layers.3.self_attn.indexer.quant_type": "BF16",
            },
            (0, 3, 6),
        )
        == ()
    )


def test_any_layer_filter_disables_global_fallback_for_unlisted_layers():
    cache = _load_indexer_cache_module()

    assert cache.get_modelslim_c8_indexer_layers(
        {
            "indexer_quant_type": "INT8_DYNAMIC",
            "model.layers.3.self_attn.indexer.quant_type": "INT8_DYNAMIC",
        },
        (0, 3, 6),
    ) == (3,)


def test_mxfp8_indexer_projection_marks_only_matching_layer():
    cache = _load_indexer_cache_module()

    assert cache.get_modelslim_c8_indexer_layers(
        {
            "model.layers.1.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
            "model.layers.2.self_attn.indexer.wq_b_weight": "W8A8_DYNAMIC",
        },
        (1, 2),
    ) == (1,)


def test_glm_freq4_offset3_filters_only_active_dynamic_indexers():
    cache = _load_indexer_cache_module()
    active_layers = (0, 1, 2, *range(6, 78, 4))
    quant_description = {
        f"model.layers.{layer_id}.self_attn.indexer.quant_type": "INT8_DYNAMIC"
        for layer_id in active_layers
    }
    quant_description["model.layers.3.self_attn.indexer.quant_type"] = "INT8"

    assert (
        cache.get_modelslim_c8_indexer_layers(quant_description, active_layers)
        == active_layers
    )
    assert cache.get_modelslim_c8_indexer_layers(quant_description, (3,)) == ()
