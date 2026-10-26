from __future__ import annotations

import torch


class PlanState:
    def __init__(self, rows: int, topk: int, capacity: int) -> None:
        self.last_req_ids = [-1] * rows
        self.slot_to_token = [[-1] * capacity for _ in range(rows)]
        self.lru_slots = [list(range(capacity)) for _ in range(rows)]
        self.current_slots = [[-1] * topk for _ in range(rows)]
        self.miss_count = [0] * rows
        self.miss_tokens = [[-1] * topk for _ in range(rows)]
        self.miss_slots = [[-1] * topk for _ in range(rows)]

    @classmethod
    def empty(cls, rows: int, topk: int, capacity: int) -> PlanState:
        return cls(rows, topk, capacity)


def _source_token_is_valid(
    token: int,
    *,
    row: int,
    visible_seq_lens: list[int],
    token_to_req: list[int],
    block_table: list[list[int]],
    max_token: int,
    block_size: int,
    host_num_blocks: int,
) -> bool:
    if token < 0 or token >= max_token or token >= visible_seq_lens[row]:
        return False
    request = token_to_req[row]
    if request < 0 or request >= len(block_table) or block_size <= 0:
        return False
    block = token // block_size
    if block >= len(block_table[request]):
        return False
    physical_block = block_table[request][block]
    return 0 <= physical_block < host_num_blocks


def plan_reference(
    state: PlanState,
    *,
    req_ids: list[int],
    topk_indices: list[list[int]],
    stable_prefix_lens: list[int],
    visible_seq_lens: list[int],
    token_to_req: list[int],
    block_table: list[list[int]],
    active_rows: int,
    max_token: int,
    block_size: int,
    host_num_blocks: int,
) -> None:
    topk = len(state.current_slots[0])
    capacity = len(state.slot_to_token[0])
    for row in range(active_rows):
        state.current_slots[row] = [-1] * topk
        state.miss_tokens[row] = [-1] * topk
        state.miss_slots[row] = [-1] * topk
        state.miss_count[row] = 0
        request = token_to_req[row]
        current_req_id = req_ids[request] if 0 <= request < len(req_ids) else -1
        if state.last_req_ids[row] != current_req_id:
            state.last_req_ids[row] = current_req_id
            state.slot_to_token[row] = [-1] * capacity
            state.lru_slots[row] = list(range(capacity))

        valid_positions: dict[int, int] = {}
        for position, token in enumerate(topk_indices[row]):
            if (
                _source_token_is_valid(
                    token,
                    row=row,
                    visible_seq_lens=visible_seq_lens,
                    token_to_req=token_to_req,
                    block_table=block_table,
                    max_token=max_token,
                    block_size=block_size,
                    host_num_blocks=host_num_blocks,
                )
                and token not in valid_positions
            ):
                valid_positions[token] = position

        stable_prefix = min(max(stable_prefix_lens[row], 0), max_token)
        evictable_slots: list[int] = []
        hit_slots: list[int] = []
        hit_owners: dict[int, int] = {}
        for order, slot in enumerate(state.lru_slots[row]):
            if not 0 <= slot < capacity:
                continue
            token = state.slot_to_token[row][slot]
            if 0 <= token < max_token and token >= stable_prefix:
                state.slot_to_token[row][slot] = -1
                token = -1
            if token in valid_positions:
                hit_slots.append(slot)
                hit_owners[token] = order
            else:
                evictable_slots.append(slot)

        for token, order in hit_owners.items():
            state.current_slots[row][valid_positions[token]] = state.lru_slots[row][
                order
            ]

        miss_positions = [
            position
            for position, token in enumerate(topk_indices[row])
            if _source_token_is_valid(
                token,
                row=row,
                visible_seq_lens=visible_seq_lens,
                token_to_req=token_to_req,
                block_table=block_table,
                max_token=max_token,
                block_size=block_size,
                host_num_blocks=host_num_blocks,
            )
            and state.current_slots[row][position] < 0
        ]
        assign_count = min(len(miss_positions), len(evictable_slots))
        for miss, position in enumerate(miss_positions[:assign_count]):
            slot = evictable_slots[miss]
            token = topk_indices[row][position]
            state.slot_to_token[row][slot] = token
            state.current_slots[row][position] = slot
            state.miss_tokens[row][miss] = token
            state.miss_slots[row][miss] = slot
        state.miss_count[row] = assign_count
        state.lru_slots[row] = (
            evictable_slots[assign_count:] + evictable_slots[:assign_count] + hit_slots
        )


def transfer_reference(
    *,
    state: PlanState,
    token_to_req: list[int],
    block_table: list[list[int]],
    active_rows: int,
    block_size: int,
    host_k: torch.Tensor,
    host_rope: torch.Tensor,
    resident_k: torch.Tensor,
    resident_rope: torch.Tensor,
) -> None:
    for row in range(active_rows):
        request = token_to_req[row]
        for miss in range(state.miss_count[row]):
            token = state.miss_tokens[row][miss]
            slot = state.miss_slots[row][miss]
            physical_block = block_table[request][token // block_size]
            physical_token = physical_block * block_size + token % block_size
            resident_k[row, slot].copy_(host_k[physical_token])
            resident_rope[row, slot].copy_(host_rope[physical_token])
