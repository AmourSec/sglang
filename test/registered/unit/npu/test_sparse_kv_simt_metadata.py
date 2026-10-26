from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
    ),
)

from metadata import (  # noqa: E402
    SparseKVMetadataInput,
    advance_stable_frontiers,
    build_request_identities,
    build_sparse_kv_metadata,
    pad_topk_rows,
    should_use_sparse_kv_simt,
    should_write_host_kv,
)


class _IdleForwardMode:
    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend_v2(self) -> bool:
        return False


def test_mtp3_metadata_uses_four_causal_rows_per_request() -> None:
    # Given: two requests, each with the four linear MTP query positions.
    generations = torch.zeros(8, dtype=torch.int64)
    identities = build_request_identities(
        generations, torch.tensor([2, 5]), torch.tensor([True, True])
    )
    positions = torch.tensor([9, 10, 11, 12, 20, 21, 22, 23])
    topk = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)

    # When: metadata is built for the linear four-query layout.
    metadata = build_sparse_kv_metadata(
        SparseKVMetadataInput(
            request_slots=torch.tensor([2, 5]),
            request_identities=identities,
            request_valid=torch.tensor([True, True]),
            row_execution_valid=torch.ones(8, dtype=torch.bool),
            query_lens=torch.tensor([4, 4]),
            positions=positions,
            topk_indices=topk,
            stable_prefix_lens=torch.tensor([9, 9, 9, 9, 20, 20, 20, 20]),
            max_context_len=32,
            block_size=4,
        )
    )

    # Then: every query has its own request mapping and causal visible length.
    assert metadata.token_to_req.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert metadata.req_ids.tolist() == identities.tolist()
    assert metadata.visible_seq_lens.tolist() == [10, 11, 12, 13, 21, 22, 23, 24]
    assert metadata.block_table.tolist() == [
        list(range(16, 24)),
        list(range(40, 48)),
    ]


def test_mtp0_keeps_a_real_length_one_request_active() -> None:
    # Given: one real request at sequence length one and one padding request.
    identities = build_request_identities(
        torch.zeros(4, dtype=torch.int64),
        torch.tensor([1, 0]),
        torch.tensor([True, False]),
    )

    # When: rows are built using an explicit validity mask.
    metadata = build_sparse_kv_metadata(
        SparseKVMetadataInput(
            request_slots=torch.tensor([1, 0]),
            request_identities=identities,
            request_valid=torch.tensor([True, False]),
            row_execution_valid=torch.ones(2, dtype=torch.bool),
            query_lens=torch.tensor([1, 1]),
            positions=torch.tensor([0, 0]),
            topk_indices=torch.tensor([[0, -1], [0, -1]], dtype=torch.int32),
            stable_prefix_lens=torch.tensor([0, 0]),
            max_context_len=8,
            block_size=4,
        )
    )

    # Then: seq_len == 1 is not treated as padding; only request_valid masks rows.
    assert metadata.row_valid.tolist() == [True, False]
    assert metadata.visible_seq_lens.tolist() == [1, 0]
    assert metadata.topk_indices.tolist() == [[0, -1], [-1, -1]]


def test_slot_reuse_changes_the_plan_request_identity() -> None:
    # Given: a request identity for pool slot three.
    generations = torch.zeros(4, dtype=torch.int64)
    before = build_request_identities(
        generations, torch.tensor([3]), torch.tensor([True])
    )

    # When: the slot is released and allocated to another request.
    generations[3] += 1
    after = build_request_identities(
        generations, torch.tensor([3]), torch.tensor([True])
    )

    # Then: persistent Plan state cannot mistake the new request for the old one.
    assert before.item() != after.item()
    assert before.item() & 0xFFFFFFFF == after.item() & 0xFFFFFFFF == 3


def test_dormant_row_invalidates_from_earliest_unseen_rewrite() -> None:
    # Given: row zero holds request 7 while another row handles its rewrite.
    stable_frontiers = torch.tensor([102, 0, 0, 0], dtype=torch.int32)
    last_req_ids = torch.tensor([7, 8, -1, -1], dtype=torch.int64)

    # When: the request is written at positions 96 and 97 away from row zero.
    advance_stable_frontiers(
        stable_frontiers,
        last_req_ids,
        request_identities=torch.tensor([8, 7]),
        request_valid=torch.tensor([True, True]),
        token_to_req=torch.tensor([1, 1], dtype=torch.int32),
        positions=torch.tensor([96, 97], dtype=torch.int32),
        row_valid=torch.tensor([True, True]),
        max_context_len=140_000,
    )

    # Then: the dormant row retains the earliest unseen rewrite frontier.
    assert stable_frontiers[0].item() == 96


def test_dp_padding_adds_invalid_topk_rows_to_match_query_rows() -> None:
    # Given: the local indexer emitted two rows while DP padded query has four.
    local_topk = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)

    # When: production top-k padding aligns it with the query tensor.
    padded_topk = pad_topk_rows(local_topk, num_rows=4)

    # Then: added rows cannot select stale host tokens.
    assert padded_topk.tolist() == [[4, 5], [6, 7], [-1, -1], [-1, -1]]


def test_simt_host_write_is_independent_of_paged_kv_save_flag() -> None:
    assert should_write_host_kv(save_kv_cache=False, uses_simt=True)
    assert not should_write_host_kv(save_kv_cache=False, uses_simt=False)
    assert should_write_host_kv(save_kv_cache=True, uses_simt=False)


def test_idle_padding_routes_to_simt_with_an_invalid_attention_row() -> None:
    # Given: eager DP idle materialized one dummy row in reserved request slot zero.
    out_cache_loc = torch.tensor([0])
    identities = build_request_identities(
        torch.zeros(2, dtype=torch.int64),
        torch.tensor([0]),
        torch.tensor([False]),
    )

    # When: the production dispatch and metadata helpers process the idle row.
    use_simt = should_use_sparse_kv_simt(_IdleForwardMode())
    metadata = build_sparse_kv_metadata(
        SparseKVMetadataInput(
            request_slots=torch.tensor([0]),
            request_identities=identities,
            request_valid=torch.tensor([False]),
            row_execution_valid=out_cache_loc.gt(0),
            query_lens=torch.tensor([1]),
            positions=torch.tensor([0]),
            topk_indices=torch.tensor([[0, -1]], dtype=torch.int32),
            stable_prefix_lens=torch.tensor([0]),
            max_context_len=8,
            block_size=4,
        )
    )

    # Then: idle uses SIMT and cannot expose the uninitialized resident padding slot.
    assert use_simt
    assert not metadata.row_valid.item()
    assert metadata.visible_seq_lens.item() == 0
    assert metadata.topk_indices.tolist() == [[-1, -1]]
