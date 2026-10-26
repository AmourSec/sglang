from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass

from ..reference import PlanState, plan_reference
from .cases import BLOCK_SIZE, TOPK, BenchmarkCase


@dataclass
class Workload:
    case: BenchmarkCase
    req_ids: list[int]
    topk_indices: list[list[int]]
    stable_prefix_lens: list[int]
    visible_seq_lens: list[int]
    token_to_req: list[int]
    block_table: list[list[int]]
    initial_state: PlanState

    @property
    def occupancy_min(self) -> int:
        return min(
            sum(token >= 0 for token in row) for row in self.initial_state.slot_to_token
        )

    @property
    def occupancy_max(self) -> int:
        return max(
            sum(token >= 0 for token in row) for row in self.initial_state.slot_to_token
        )


def build_workload(case: BenchmarkCase) -> Workload:
    req_ids = [10000 + request for request in range(case.batch_size)]
    topk_indices: list[list[int]] = []
    state = PlanState.empty(case.rows, TOPK, case.capacity)
    for row in range(case.rows):
        rng = random.Random(case.seed + row * 1009)
        query = rng.sample(range(case.seq_len), TOPK)
        topk_indices.append(query)
        cached = rng.sample(query, case.hits_per_row)
        excluded = set(query)
        fillers = (token for token in range(case.seq_len) if token not in excluded)
        values = cached + list(
            itertools.islice(fillers, max(0, case.capacity - len(cached)))
        )
        values.extend([-1] * (case.capacity - len(values)))
        rng.shuffle(values)
        for slot, token in enumerate(values[: case.capacity]):
            state.slot_to_token[row][slot] = token
        rng.shuffle(state.lru_slots[row])
        state.last_req_ids[row] = req_ids[row // case.width]

    pages = case.pages_per_request
    block_table = []
    for request in range(case.batch_size):
        start = (case.batch_size - request - 1) * pages
        block_table.append([start + pages - page - 1 for page in range(pages)])
    token_to_req = [row // case.width for row in range(case.rows)]
    visible = [case.seq_len + row % case.width + 1 for row in range(case.rows)]
    return Workload(
        case,
        req_ids,
        topk_indices,
        [case.seq_len] * case.rows,
        visible,
        token_to_req,
        block_table,
        state,
    )


def oracle_post_state(workload: Workload) -> PlanState:
    result = copy.deepcopy(workload.initial_state)
    case = workload.case
    plan_reference(
        result,
        req_ids=workload.req_ids,
        topk_indices=workload.topk_indices,
        stable_prefix_lens=workload.stable_prefix_lens,
        visible_seq_lens=workload.visible_seq_lens,
        token_to_req=workload.token_to_req,
        block_table=workload.block_table,
        active_rows=case.rows,
        max_token=case.seq_len + case.width,
        block_size=BLOCK_SIZE,
        host_num_blocks=case.host_num_blocks,
    )
    return result
