"""
Einsum NVQIR Python Integration

This package provides Python bindings and utilities for capturing
quantum circuits from CUDA-Q kernels and converting them to tensor
network (einsum) representations.

Modules:
    einsum_sidecar: ctypes client for C++ sidecar API
    einsum_circuit: EinsumCircuit data class for circuit representation
    cudaq_einsum: High-level API for circuit capture

Quick start:
    from cudaq_einsum import capture_circuit

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    circuit = capture_circuit(bell)
    state = circuit.get_state_vector()
"""

import os
import shutil
from typing import Optional

__version__ = "0.1.0"

from .einsum_sidecar import EinsumSidecar
from .einsum_circuit import EinsumCircuit, ParametricGateInfo, PARAMETRIC_GATE_CODES
from .cudaq_einsum import capture_circuit
from .batched import make_rotation_matrices, build_batched_args, batched_contract

try:
    from .multistream import (
        multistream_batched_contract,
        prepare_multistream_backend,
        run_prepared_multistream,
    )
except Exception as exc:
    def _raise_multistream_unavailable(*args, **kwargs):
        raise RuntimeError(
            "The optional multi-stream backend is unavailable in this environment. "
            "Build the C++ extension and ensure its Python dependencies are installed."
        ) from exc

    multistream_batched_contract = _raise_multistream_unavailable
    prepare_multistream_backend = _raise_multistream_unavailable
    run_prepared_multistream = _raise_multistream_unavailable


def _find_cudaq_root() -> Optional[str]:
    """
    Auto-detect the CUDA-Q installation root.

    Search order:
      1. CUDAQ_ROOT environment variable
      2. Active conda environment ($CONDA_PREFIX)
      3. pip-installed cudaq (via import)
    """
    from pathlib import Path

    # 1. Explicit env var
    root = os.environ.get("CUDAQ_ROOT")
    if root and Path(root).is_dir():
        return root

    # 2. Conda prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for lib_name in ("libnvqir.so", "libcudaq.so"):
            if Path(conda_prefix, "lib", lib_name).exists():
                return conda_prefix

    # 3. pip-installed cudaq
    try:
        import cudaq  # noqa: F401

        cudaq_file = Path(cudaq.__file__)
        site_packages_dir = cudaq_file.parent.parent
        if (site_packages_dir / "lib" / "libnvqir.so").exists():
            return str(site_packages_dir)
            
        for candidate in [
            cudaq_file.parent,
            site_packages_dir,
            site_packages_dir.parent,
        ]:
            for lib_dir in ["lib", "cudaq/lib", "../lib"]:
                for lib_name in ("libnvqir.so", "libcudaq.so"):
                    if (candidate / lib_dir / lib_name).exists():
                        return str((candidate / lib_dir).parent.resolve())
    except ImportError:
        pass

    return None


def install_cudaq_target(cudaq_root: Optional[str] = None) -> str:
    """
    Install the einsum simulator into the CUDA-Q installation.

    This copies ``libnvqir-einsum.so`` and ``einsum.yml`` from the bundled
    package data into the CUDA-Q lib and targets directories, making
    ``cudaq.set_target('einsum')`` work.

    Must be called once after ``pip install .``.

    Args:
        cudaq_root: Path to the CUDA-Q installation root. If None, the
                    root is auto-detected (CUDAQ_ROOT env var, conda env,
                    or pip-installed cudaq).

    Returns:
        The cudaq_root path where files were installed.

    Raises:
        FileNotFoundError: If the bundled .so is not found (package not
                           properly installed).
        RuntimeError: If CUDA-Q installation cannot be located.

    Example:
        import cudaq_einsum
        cudaq_einsum.install_cudaq_target()
        # or with an explicit path:
        # cudaq_einsum.install_cudaq_target("/path/to/cudaq")
    """
    from pathlib import Path

    pkg_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    so_path = pkg_dir / "libnvqir-einsum.so"
    yml_path = pkg_dir / "einsum.yml"

    if not so_path.exists():
        raise FileNotFoundError(
            f"Bundled library not found: {so_path}\n"
            "The package may not have been built correctly.\n"
            "Try: pip install . --force-reinstall"
        )

    if cudaq_root is None:
        cudaq_root = _find_cudaq_root()
    if cudaq_root is None:
        raise RuntimeError(
            "Cannot find CUDA-Q installation.\n"
            "Options:\n"
            "  1. Set env var:  export CUDAQ_ROOT=/path/to/cudaq\n"
            "  2. Conda env:    conda activate cudaq-env\n"
            "  3. pip install:  pip install cuda-quantum"
        )

    lib_dir = Path(cudaq_root) / "lib"
    targets_dir = Path(cudaq_root) / "targets"
    lib_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(so_path), str(lib_dir / "libnvqir-einsum.so"))
    print(f"Installed: {lib_dir / 'libnvqir-einsum.so'}")

    if yml_path.exists():
        shutil.copy2(str(yml_path), str(targets_dir / "einsum.yml"))
        print(f"Installed: {targets_dir / 'einsum.yml'}")

    print(f"CUDA-Q target 'einsum' installed to: {cudaq_root}")
    return cudaq_root


__all__ = [
    "EinsumSidecar",
    "EinsumCircuit",
    "ParametricGateInfo",
    "PARAMETRIC_GATE_CODES",
    "capture_circuit",
    "install_cudaq_target",
    "make_rotation_matrices",
    "build_batched_args",
    "batched_contract",
    "multistream_batched_contract",
    "prepare_multistream_backend",
    "run_prepared_multistream",
]
