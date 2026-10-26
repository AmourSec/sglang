from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
ASCEND_HOME = Path(
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
)

try:
    import torch_npu
except ImportError as error:
    message = "torch_npu 2.10 is required to build sgl-kernel-npu-sparse-kv"
    raise RuntimeError(message) from error

TORCH_NPU_HOME = Path(torch_npu.__file__).resolve().parent

extension = CppExtension(
    name="sgl_kernel_npu_sparse_kv._C",
    sources=[
        str(ROOT / "csrc" / "binding.cpp"),
        str(ROOT / "aclnn_torch_adapter" / "NPUBridge.cpp"),
        str(ROOT / "aclnn_torch_adapter" / "NPUStorageImpl.cpp"),
    ],
    include_dirs=[
        str(TORCH_NPU_HOME / "include"),
        str(ASCEND_HOME / "include"),
        str(ROOT),
    ],
    library_dirs=[
        str(TORCH_NPU_HOME / "lib"),
        str(ASCEND_HOME / "lib64"),
    ],
    libraries=["torch_npu", "ascendcl", "dl"],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
)

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
