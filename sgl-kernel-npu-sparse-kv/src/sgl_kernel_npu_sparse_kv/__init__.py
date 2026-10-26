from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path
from typing import Final

_VENDOR_NAME: Final = "sgl_sparse_kv_transformer"
_CUSTOM_OP_API: ctypes.CDLL | None = None


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if value not in entries:
        os.environ[name] = ":".join((value, *entries))


def ensure_loaded() -> None:
    """Load the packaged custom OPP and register ``torch.ops.sgl_sparse_kv``."""
    if "sgl_kernel_npu_sparse_kv._C" in sys.modules:
        return

    global _CUSTOM_OP_API

    package_dir = Path(__file__).resolve().parent
    vendor_dir = package_dir / "_cann_ops_custom" / "vendors" / _VENDOR_NAME
    if not vendor_dir.is_dir():
        message = (
            f"custom OPP is missing at {vendor_dir}; build this package on an "
            "Ascend 950 host with `bash build.sh`"
        )
        raise RuntimeError(message)

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
    vendor_lib = vendor_dir / "op_api" / "lib"
    custom_op_api = vendor_lib / "libcust_opapi.so"
    if not custom_op_api.is_file():
        raise RuntimeError(f"packaged custom op API is missing at {custom_op_api}")
    _prepend_env_path("LD_LIBRARY_PATH", vendor_lib)
    importlib.import_module("torch_npu")
    # The adapter resolves ACLNN symbols with dlsym on this library's handle.
    # Global visibility can interpose CANN symbols used by MemFabric at teardown.
    _CUSTOM_OP_API = ctypes.CDLL(str(custom_op_api), mode=ctypes.RTLD_LOCAL)
    importlib.import_module("sgl_kernel_npu_sparse_kv._C")


__all__ = ["ensure_loaded"]
