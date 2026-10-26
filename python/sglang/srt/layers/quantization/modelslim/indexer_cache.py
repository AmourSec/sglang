from __future__ import annotations

from collections.abc import Mapping, Sequence

_LAYER_SUFFIXES = (".indexer.quant_type", ".indexer.wq_b_weight")
_DYNAMIC_INDEXER_TYPES = frozenset({"INT8_DYNAMIC", "W8A8_MXFP8"})


def get_modelslim_c8_indexer_layers(
    quant_description: Mapping[str, str], layer_ids: Sequence[int]
) -> tuple[int, ...]:
    has_layer_filter = any(name.endswith(_LAYER_SUFFIXES) for name in quant_description)
    if not has_layer_filter:
        return tuple(layer_ids)

    c8_layers = []
    for layer_id in layer_ids:
        layer_marker = f".layers.{layer_id}."
        is_dynamic = any(
            layer_marker in f".{name}"
            and name.endswith(_LAYER_SUFFIXES)
            and value in _DYNAMIC_INDEXER_TYPES
            for name, value in quant_description.items()
        )
        if is_dynamic:
            c8_layers.append(layer_id)
    return tuple(c8_layers)
