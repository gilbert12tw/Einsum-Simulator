#!/usr/bin/env python3
from setuptools import setup, Extension
import os

try:
    import pybind11
except ImportError as e:
    raise RuntimeError("pybind11 is required. Install with: pip install pybind11") from e

CUDA_ROOT = os.environ.get(
    "CUDA_ROOT",
    "/usr/local/cuda",
)
CUTENSOR_ROOT = os.environ.get(
    "CUTENSOR_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cutensor"),
)

# CUDA_ROOT and CUTENSOR_ROOT are intentionally environment-driven so this
# bundle can be moved between machines without editing setup.py. CUTENSOR_ROOT
# should point to the unpacked NVIDIA cuTENSOR archive root.
cutensor_lib_candidates = [
    os.path.join(CUTENSOR_ROOT, "lib", "12"),
    os.path.join(CUTENSOR_ROOT, "lib"),
]
for candidate in cutensor_lib_candidates:
    if os.path.isdir(candidate):
        cutensor_lib = candidate
        break
else:
    cutensor_lib = cutensor_lib_candidates[0]

cuda_lib = os.path.join(CUDA_ROOT, "lib64")


ext_modules = [
    Extension(
        "qsvm_cutensor_backend",
        ["qsvm_cutensor_backend.cpp"],
        include_dirs=[
            pybind11.get_include(),
            os.path.join(CUDA_ROOT, "include"),
            os.path.join(CUTENSOR_ROOT, "include"),
        ],
        library_dirs=[cutensor_lib, cuda_lib],
        libraries=["cutensor", "cudart", "cublas"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++14"],
        extra_link_args=[f"-Wl,-rpath,{cutensor_lib}:{cuda_lib}"],
    )
]

setup(
    name="qsvm_cutensor_backend",
    version="0.1.0",
    description="QSVM cuTENSOR multi-stream contraction backend",
    ext_modules=ext_modules,
)
