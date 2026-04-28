"""
Batched evaluation of parameterized quantum circuits via tensor network contraction.

Provides functions to evaluate a circuit across B different parameter sets
in a single einsum contraction by adding a batch dimension to parametric gates.

Functions:
    make_rotation_matrices: Build (B, num_params, 2, 2) rotation matrices.
    build_batched_args: Build sublist-format einsum args with batch dimension.
    batched_contract: Convenience wrapper that calls build_batched_args + contract.
"""

import numpy as np
from typing import List, Optional, Any


def make_rotation_matrices(thetas, codes, backend='numpy', device=None):
    """
    Vectorized rotation-matrix builder for rx, ry, rz, and p gates.

    Args:
        thetas: (B, num_params) array of rotation angles.
        codes: (num_params,) array of gate type codes — rx=0, ry=1, rz=2, p/r1=3.
        backend: 'numpy' or 'torch'.
        device: Device for torch tensors (e.g. 'cuda'). Ignored for numpy.

    Returns:
        (B, num_params, 2, 2) complex128 array of 2x2 rotation matrices.
    """
    if backend == 'torch':
        return _make_rotation_matrices_torch(thetas, codes, device)
    else:
        return _make_rotation_matrices_numpy(thetas, codes)


def _make_rotation_matrices_numpy(thetas, codes):
    """NumPy implementation of rotation matrix builder."""
    thetas = np.asarray(thetas, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)

    orig_shape = thetas.shape  # (B, num_params)
    flat = thetas.reshape(-1)  # (N,)
    N = flat.size

    half = flat * 0.5
    cosv = np.cos(half).astype(np.complex128)
    sinv = np.sin(half).astype(np.complex128)
    phase = np.exp(-1j * half)
    conjp = phase.conj()
    phaseP = np.exp(1j * flat.astype(np.complex128))

    reps = N // codes.size
    cf = np.tile(codes, reps)

    is_rx = cf == 0
    is_ry = cf == 1
    is_rz = cf == 2
    is_p = cf == 3

    out = np.empty((N, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = np.where(is_p, 1.0 + 0j, np.where(is_rz, phase, cosv))
    out[:, 1, 1] = np.where(is_p, phaseP, np.where(is_rz, conjp, cosv))
    out[:, 0, 1] = np.where(is_rx, -1j * sinv, np.where(is_ry, -sinv, 0.0 + 0j))
    out[:, 1, 0] = np.where(is_rx, -1j * sinv, np.where(is_ry, sinv, 0.0 + 0j))

    return out.reshape(*orig_shape, 2, 2)


def _make_rotation_matrices_torch(thetas, codes, device=None):
    """PyTorch implementation of rotation matrix builder."""
    import torch

    if not isinstance(thetas, torch.Tensor):
        thetas = torch.tensor(thetas, dtype=torch.float64, device=device)
    else:
        thetas = thetas.to(dtype=torch.float64)
        if device is not None:
            thetas = thetas.to(device=device)

    if not isinstance(codes, torch.Tensor):
        codes = torch.tensor(codes, dtype=torch.long, device=thetas.device)
    else:
        codes = codes.to(dtype=torch.long, device=thetas.device)

    orig_device = thetas.device
    work_device = torch.device("cpu") if orig_device.type == "cuda" else orig_device
    thetas = thetas.to(work_device)
    codes = codes.to(work_device)

    orig_shape = thetas.shape
    flat = thetas.reshape(-1)
    N = flat.numel()

    half = flat * 0.5
    cosv = torch.cos(half).to(torch.complex128)
    sinv = torch.sin(half).to(torch.complex128)
    phase = torch.exp(-1j * half)
    conjp = phase.conj()
    phaseP = torch.exp(1j * flat.to(torch.complex128))

    reps = N // codes.numel()
    cf = codes.repeat(reps).to(flat.device)

    is_rx = cf == 0
    is_ry = cf == 1
    is_rz = cf == 2
    is_p = cf == 3

    out = torch.empty((N, 2, 2), dtype=torch.complex128, device=flat.device)
    out[:, 0, 0] = torch.where(is_p, 1.0 + 0j, torch.where(is_rz, phase, cosv))
    out[:, 1, 1] = torch.where(is_p, phaseP, torch.where(is_rz, conjp, cosv))
    out[:, 0, 1] = torch.where(is_rx, -1j * sinv, torch.where(is_ry, -sinv, 0.0 + 0j))
    out[:, 1, 0] = torch.where(is_rx, -1j * sinv, torch.where(is_ry, sinv, 0.0 + 0j))

    return out.to(orig_device).view(*orig_shape, 2, 2)


def build_batched_args(
    circuit,
    param_positions: List[int],
    batch_rot_mats,
    mode: str = 'amplitude',
    target_state: Optional[int] = None,
):
    """
    Build sublist-format einsum args with a batch dimension for parametric gates.

    Args:
        circuit: EinsumCircuit captured with dummy parameters.
        param_positions: List of gate indices that are parametric.
        batch_rot_mats: (B, num_params, 2, 2) rotation matrices (numpy or torch).
        mode: 'amplitude' — project onto target state, output is (B,) complex.
              'statevector' — keep output indices, output is (B, 2, 2, ..., 2).
        target_state: Integer for the target basis state in amplitude mode.
                      Defaults to 0 (|0...0⟩).

    Returns:
        List of args suitable for np.einsum(*args) or torch.einsum(*args).
    """
    if target_state is None:
        target_state = 0

    if len(param_positions) == 0:
        raise ValueError(
            "param_positions is empty. build_batched_args requires at least one "
            "parametric gate to define the batch dimension. "
            "Use circuit.to_einsum_sublist_args() for non-parametric circuits."
        )

    expected_params = len(param_positions)
    actual_params = batch_rot_mats.shape[1] if hasattr(batch_rot_mats, 'shape') else len(batch_rot_mats[0])
    if actual_params != expected_params:
        raise ValueError(
            f"Shape mismatch: batch_rot_mats has {actual_params} parameter slots "
            f"but param_positions has {expected_params} entries."
        )

    is_torch = _is_torch_tensor(batch_rot_mats)
    batch_idx = circuit.max_index + 1
    param_set = set(param_positions)

    if is_torch:
        import torch
        dev = batch_rot_mats.device
        zero_ket = torch.tensor([1.0, 0.0], dtype=torch.complex128, device=dev)
        one_ket = torch.tensor([0.0, 1.0], dtype=torch.complex128, device=dev)

        def to_tensor(np_arr):
            return torch.tensor(np_arr, dtype=torch.complex128, device=dev)
    else:
        zero_ket = np.array([1.0, 0.0], dtype=np.complex128)
        one_ket = np.array([0.0, 1.0], dtype=np.complex128)

        def to_tensor(np_arr):
            return np.asarray(np_arr, dtype=np.complex128)

    args = []

    # Initial |0⟩ kets
    for idx in circuit.initial_indices:
        args.append(zero_ket)
        args.append([idx])

    # Gate tensors
    param_count = 0
    for i, (tensor, indices) in enumerate(zip(circuit.gate_tensors, circuit.gate_indices)):
        if i in param_set:
            args.append(batch_rot_mats[:, param_count])  # (B, 2, 2)
            args.append([batch_idx] + list(indices))
            param_count += 1
        else:
            args.append(to_tensor(tensor))
            args.append(list(indices))

    # Output
    if mode == 'amplitude':
        # Project onto target basis state
        bra_vectors = _basis_bras(
            circuit.num_qubits, target_state, zero_ket, one_ket
        )
        for idx, bra in zip(circuit.output_indices, bra_vectors):
            args.append(bra)
            args.append([idx])
        args.append([batch_idx])
    elif mode == 'statevector':
        args.append([batch_idx] + list(circuit.output_indices))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'amplitude' or 'statevector'.")

    return args


def batched_contract(
    circuit,
    param_positions: List[int],
    batch_rot_mats,
    mode: str = 'amplitude',
    target_state: Optional[int] = None,
    optimize: bool = True,
):
    """
    Convenience function: build batched args and contract.

    Args:
        circuit: EinsumCircuit captured with dummy parameters.
        param_positions: List of gate indices that are parametric.
        batch_rot_mats: (B, num_params, 2, 2) rotation matrices.
        mode: 'amplitude' or 'statevector'.
        target_state: Basis state index for amplitude mode (default 0).
        optimize: Passed to np.einsum / opt_einsum / cuquantum.contract.

    Returns:
        Contracted result — (B,) complex for amplitude mode,
        (B, 2, ..., 2) for statevector mode.
    """
    args = build_batched_args(
        circuit, param_positions, batch_rot_mats,
        mode=mode, target_state=target_state,
    )

    is_torch = _is_torch_tensor(batch_rot_mats)

    if is_torch:
        try:
            from cuquantum import contract
            cuq_optimize = None if optimize is True else optimize
            return contract(*args, optimize=cuq_optimize)
        except ImportError:
            import torch
            return torch.einsum(*args)
    else:
        try:
            import opt_einsum
            return opt_einsum.contract(*args)
        except ImportError:
            return np.einsum(*args, optimize=optimize)


def _is_torch_tensor(obj) -> bool:
    """Check if obj is a torch.Tensor without importing torch."""
    return type(obj).__module__.startswith('torch')


def _basis_bras(num_qubits, state_index, zero_ket, one_ket):
    """Return per-qubit bra vectors for a computational basis state."""
    bras = []
    for i in range(num_qubits):
        # MSB convention: q[0] is the most significant bit
        bit = (state_index >> (num_qubits - 1 - i)) & 1
        bras.append(one_ket if bit else zero_ket)
    return bras
