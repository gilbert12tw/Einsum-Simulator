"""
High-level wrappers for the optional cuTENSOR multi-stream backend.
"""

from typing import Optional

try:
    from .qsvm_cudaq_cpp_backend import prepare_cpp_backend, run_cpp_backend
except ImportError:
    from qsvm_cudaq_cpp_backend import prepare_cpp_backend, run_cpp_backend


def _ensure_rotation_tensor(rotation_matrices):
    import torch

    if not torch.is_tensor(rotation_matrices):
        rotation_matrices = torch.as_tensor(
            rotation_matrices, dtype=torch.complex128
        )
    elif rotation_matrices.dtype not in (torch.complex64, torch.complex128):
        rotation_matrices = rotation_matrices.to(torch.complex128)
    return rotation_matrices


def prepare_multistream_backend(
    captured_circuit,
    param_positions,
    rotation_matrices,
    *,
    require_cuda: bool = True,
    target_state: Optional[int] = None,
    module_name: str = "qsvm_cutensor_backend",
    verbose: bool = False,
):
    rotation_matrices = _ensure_rotation_tensor(rotation_matrices)
    if require_cuda and rotation_matrices.device.type != "cuda":
        raise RuntimeError(
            "prepare_multistream_backend requires rotation_matrices to be a CUDA tensor. "
            "Move it with rotation_matrices.to('cuda')."
        )

    return prepare_cpp_backend(
        captured_circuit,
        param_positions,
        rotation_matrices,
        target_state=target_state,
        module_name=module_name,
        verbose=verbose,
    )


def run_prepared_multistream(prepared):
    return run_cpp_backend(prepared)


def multistream_batched_contract(
    captured_circuit,
    param_positions,
    rotation_matrices,
    *,
    require_cuda: bool = True,
    target_state: Optional[int] = None,
    module_name: str = "qsvm_cutensor_backend",
    verbose: bool = False,
    return_info: bool = False,
):
    rotation_matrices = _ensure_rotation_tensor(rotation_matrices)
    prepared = prepare_multistream_backend(
        captured_circuit,
        param_positions,
        rotation_matrices,
        require_cuda=require_cuda,
        target_state=target_state,
        module_name=module_name,
        verbose=verbose,
    )
    amplitudes = run_prepared_multistream(prepared)

    if not return_info:
        return amplitudes

    info = {
        "config": prepared["config"],
        "batch_size": prepared["batch_size"],
        "num_param_gates": int(rotation_matrices.shape[1]),
        "device": str(rotation_matrices.device),
        "dtype": str(rotation_matrices.dtype),
        "module_name": module_name,
        "target_state": 0 if target_state is None else int(target_state),
    }
    return amplitudes, info
