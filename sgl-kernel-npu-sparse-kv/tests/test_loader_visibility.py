from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_loader_keeps_native_symbols_local_and_resolves_them_by_handle(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("cc")
    flags = {"darwin": ["-dynamiclib"], "linux": ["-shared", "-fPIC"]}.get(sys.platform)
    if compiler is None or flags is None:
        pytest.skip("native loader visibility test requires a Linux/macOS C compiler")

    package = tmp_path / "sgl_kernel_npu_sparse_kv"
    vendor = package / "_cann_ops_custom/vendors/sgl_sparse_kv_transformer"
    library = vendor / "op_api/lib/libcust_opapi.so"
    library.parent.mkdir(parents=True)
    source = Path(__file__).parents[1] / "src/sgl_kernel_npu_sparse_kv/__init__.py"
    shutil.copyfile(source, package / "__init__.py")
    probe = tmp_path / "probe.c"
    probe.write_text("int sgl_sparse_kv_visibility_probe(void) { return 73; }\n")
    compiled = subprocess.run(
        [compiler, *flags, str(probe), "-o", str(library)],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr

    # Only the unavailable NPU Python imports are substituted. The production
    # loader and both dlopen/dlsym visibility checks use a real native library.
    script = r"""
import ctypes
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import sgl_kernel_npu_sparse_kv as loader

sys.modules["torch_npu"] = types.ModuleType("torch_npu")
real_import = importlib.import_module
imports = []

def import_native(name, package=None):
    imports.append(name)
    if name == "sgl_kernel_npu_sparse_kv._C":
        assert loader._CUSTOM_OP_API is not None
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module
    return real_import(name, package)

loader.importlib.import_module = import_native
loader.ensure_loaded()
assert imports == ["torch_npu", "sgl_kernel_npu_sparse_kv._C"]
handle = loader._CUSTOM_OP_API
assert handle.sgl_sparse_kv_visibility_probe() == 73
library = Path(handle._name)
assert library.is_absolute()
assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0] == str(library.parents[2])
assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == str(library.parent)

# The C++ adapter also opens a handle with RTLD_LAZY and uses dlsym(handle).
adapter_handle = ctypes.CDLL(str(library), mode=os.RTLD_LAZY | ctypes.RTLD_LOCAL)
assert adapter_handle.sgl_sparse_kv_visibility_probe() == 73
try:
    ctypes.CDLL(None).sgl_sparse_kv_visibility_probe
except AttributeError:
    print("handle lookup=73; custom OPP symbol absent from global lookup")
else:
    raise AssertionError("custom OPP symbol leaked into the global namespace")

loader.ensure_loaded()
assert loader._CUSTOM_OP_API is handle
assert imports == ["torch_npu", "sgl_kernel_npu_sparse_kv._C"]
print("loader remains idempotent and retains the native handle")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
