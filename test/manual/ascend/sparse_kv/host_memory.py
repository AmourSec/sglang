from __future__ import annotations

import torch

_POOL_INITIALIZED = False
_POOL_TOKENS = 0
_HOST_K: torch.Tensor | None = None
_HOST_ROPE: torch.Tensor | None = None


class HostPlanes:
    def __init__(self, tokens: int) -> None:
        global _HOST_K, _HOST_ROPE, _POOL_INITIALIZED, _POOL_TOKENS

        from memfabric_hybrid import offload

        required = tokens * (512 + 64) * torch.bfloat16.itemsize
        allocation = max(8 * 1024 * 1024, required + 16 * 1024 * 1024)
        if not _POOL_INITIALIZED:
            config = offload.OffloadConfig()
            config.device_id = torch.npu.current_device()
            config.reserve_size = allocation
            config.alloc_size = allocation
            config.world_size = 1
            config.rank_id = 0
            config.scene = offload.Scene.LOCAL
            config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
            result = offload.initialize(config)
            if result != 0:
                raise RuntimeError(
                    f"MemFabric LOCAL giant-page initialization failed: ret={result}"
                )
            _POOL_INITIALIZED = True
        if _HOST_K is None or _HOST_ROPE is None:
            _HOST_K = offload.empty(
                [tokens, 1, 512], dtype=torch.bfloat16, pin_memory=True
            )
            _HOST_ROPE = offload.empty(
                [tokens, 1, 64], dtype=torch.bfloat16, pin_memory=True
            )
            _POOL_TOKENS = tokens
        if tokens > _POOL_TOKENS:
            raise RuntimeError(
                "MemFabric pool is already fixed below the requested size; "
                "run the full-host test first"
            )
        self.k = _HOST_K[:tokens]
        self.rope = _HOST_ROPE[:tokens]
        if self.k.device.type != "cpu" or self.rope.device.type != "cpu":
            raise RuntimeError("MemFabric host planes must be CPU tensors")
        self.k_dva = int(offload.get_dva(self.k.data_ptr()))
        self.rope_dva = int(offload.get_dva(self.rope.data_ptr()))
        if self.k_dva == 0 or self.rope_dva == 0:
            raise RuntimeError(
                "MemFabric get_dva returned zero for a LOCAL giant-page host plane"
            )

    def bases(self) -> torch.Tensor:
        return torch.tensor(
            [self.k_dva, self.rope_dva], dtype=torch.int64, device="npu"
        )
