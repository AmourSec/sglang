from __future__ import annotations

import importlib
import inspect


def test_package_import_has_explicit_loader() -> None:
    package = importlib.import_module("sgl_kernel_npu_sparse_kv")

    assert inspect.signature(package.ensure_loaded).return_annotation in {
        "None",
        None,
    }


def test_loader_does_not_depend_on_vllm() -> None:
    package = importlib.import_module("sgl_kernel_npu_sparse_kv")

    source = inspect.getsource(package)
    assert "import vllm" not in source
    assert "from vllm" not in source


def test_loader_opens_packaged_op_api_by_absolute_path() -> None:
    package = importlib.import_module("sgl_kernel_npu_sparse_kv")

    source = inspect.getsource(package)
    assert "ctypes.CDLL" in source
    assert 'vendor_lib / "libcust_opapi.so"' in source
