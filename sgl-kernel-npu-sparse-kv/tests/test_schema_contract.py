from __future__ import annotations

import re
from pathlib import Path

import torch


def _schemas() -> dict[str, torch.FunctionSchema]:
    source = Path(__file__).parents[1].joinpath("csrc/binding.cpp").read_text()
    schemas: dict[str, torch.FunctionSchema] = {}
    for definition in re.findall(r"ops\.def\((.*?)\);", source, flags=re.DOTALL):
        fragments = re.findall(r'"((?:[^"\\]|\\.)*)"', definition)
        schema_text = "".join(
            bytes(fragment, "utf-8").decode("unicode_escape") for fragment in fragments
        )
        schema = torch._C.parse_schema(schema_text)
        schemas[schema.name] = schema
    return schemas


def _writes(schema: torch.FunctionSchema) -> list[str]:
    return [
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    ]


def test_torch_2_10_schemas_and_mutation_aliases() -> None:
    schemas = _schemas()

    assert set(schemas) == {"plan", "transfer", "plan_transfer"}
    assert _writes(schemas["plan"]) == [
        "last_req_ids",
        "slot_to_token",
        "lru_slots",
        "current_slots",
        "miss_count",
        "miss_tokens",
        "miss_slots",
        "compact_workspace",
    ]
    assert _writes(schemas["transfer"]) == ["resident_k", "resident_v"]
    assert _writes(schemas["plan_transfer"]) == [
        "last_req_ids",
        "slot_to_token",
        "lru_slots",
        "current_slots",
        "miss_count",
        "miss_tokens",
        "miss_slots",
        "compact_workspace",
        "resident_k",
        "resident_v",
    ]
