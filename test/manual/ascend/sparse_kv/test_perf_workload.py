from __future__ import annotations

import copy

import pytest

from .perf.cases import TOPK, BenchmarkCase, make_cases
from .perf.workload import build_workload, oracle_post_state
from .reference import plan_reference


def test_case_matrix_covers_mtp_and_large_batch() -> None:
    cases = make_cases([1, 32], [0, 1, 2, 3], [140000], [0.0, 1.0])
    assert len(cases) == 16
    assert {case.rows for case in cases} == {1, 2, 3, 4, 32, 64, 96, 128}


@pytest.mark.parametrize("ratio", [0.0, 0.3333, 0.375, 1.0])
def test_hit_ratio_is_exact_and_query_tokens_are_unique(ratio: float) -> None:
    case = BenchmarkCase(2, 3, 4096, ratio, capacity=2048)
    workload = build_workload(case)
    assert all(
        len(row) == TOPK and len(set(row)) == TOPK for row in workload.topk_indices
    )
    for row, query in zip(workload.initial_state.slot_to_token, workload.topk_indices):
        assert len(set(token for token in row if token >= 0)) == sum(
            token >= 0 for token in row
        )
        assert len(set(row).intersection(query)) == case.hits_per_row
    assert workload.occupancy_min == workload.occupancy_max == 2048
    assert oracle_post_state(workload).miss_count == [case.misses_per_row] * case.rows
    if 0 < ratio < 1:
        cached = set(workload.initial_state.slot_to_token[0])
        assert set(workload.topk_indices[0][: case.hits_per_row]) != cached


def test_workload_mtp_rows_page_mapping_and_max_shape() -> None:
    case = BenchmarkCase(32, 3, 143360, 0.5)
    workload = build_workload(case)
    assert len(workload.topk_indices) == 128
    assert workload.token_to_req == [row // 4 for row in range(128)]
    assert all(
        length == 143360 + row % 4 + 1
        for row, length in enumerate(workload.visible_seq_lens)
    )
    assert all(
        0 <= page < case.host_num_blocks
        for request_pages in workload.block_table
        for page in request_pages
    )


def test_queries_are_stable_across_hit_ratio_and_capacity() -> None:
    first = build_workload(BenchmarkCase(2, 2, 8192, 0.0, capacity=2048))
    second = build_workload(BenchmarkCase(2, 2, 8192, 1.0, capacity=8192))
    assert first.topk_indices == second.topk_indices


def test_oracle_second_call_is_all_hits_and_restore_replays_misses() -> None:
    case = BenchmarkCase(1, 0, 8192, 0.5)
    workload = build_workload(case)
    first = oracle_post_state(workload)
    plan_reference(
        first,
        req_ids=workload.req_ids,
        topk_indices=workload.topk_indices,
        stable_prefix_lens=workload.stable_prefix_lens,
        visible_seq_lens=workload.visible_seq_lens,
        token_to_req=workload.token_to_req,
        block_table=workload.block_table,
        active_rows=case.rows,
        max_token=case.seq_len + case.width,
        block_size=128,
        host_num_blocks=case.host_num_blocks,
    )
    assert first.miss_count == [0]
    replay = oracle_post_state(workload)
    assert replay.miss_count == [case.misses_per_row]


def test_capacity_shortage_is_reported_without_fake_entries() -> None:
    case = BenchmarkCase(1, 0, 4096, 0.0, capacity=8192)
    workload = build_workload(case)
    assert workload.occupancy_min == 2048
    assert workload.occupancy_max == 2048
    assert sum(token >= 0 for token in workload.initial_state.slot_to_token[0]) == 2048


def test_oracle_state_is_repeatable_after_restore() -> None:
    case = BenchmarkCase(1, 0, 8192, 0.5)
    workload = build_workload(case)
    initial = copy.deepcopy(vars(workload.initial_state))
    first = oracle_post_state(workload)
    second = oracle_post_state(workload)
    assert vars(first) == vars(second)
    assert first.miss_count == second.miss_count == [case.misses_per_row]
    assert vars(workload.initial_state) == initial


@pytest.mark.parametrize(
    "bad", [(0, 0, 4096, 0.5), (1, 4, 4096, 0.5), (1, 0, 2047, 0.5), (1, 0, 4096, 1.1)]
)
def test_case_validation(bad: tuple[int, int, int, float]) -> None:
    with pytest.raises(ValueError):
        BenchmarkCase(*bad)


@pytest.mark.parametrize(
    "bad",
    [
        (1, 0, 143361, 0.5),
        (33, 0, 4096, 0.5),
        (1, 0, 4096, float("nan")),
        (1, 0, 4096, float("inf")),
    ],
)
def test_case_boundary_validation(bad: tuple[int, int, int, float]) -> None:
    with pytest.raises(ValueError):
        BenchmarkCase(*bad)


def test_make_cases_rejects_duplicate_axes() -> None:
    with pytest.raises(ValueError):
        make_cases([1, 1], [0], [4096], [0.5])
    with pytest.raises(ValueError):
        make_cases([], [0], [4096], [0.5])
    with pytest.raises(ValueError):
        make_cases([1], [0], [4096], [0.5], capacities=[1024])
