#!/usr/bin/env python3
"""
Example 11: Batched QSVM Kernel Matrix via Tensor Network

This example demonstrates computing a QSVM quantum kernel matrix using the
batched evaluation API.  Instead of running one circuit per data pair, a
single einsum contraction evaluates all pairs simultaneously by adding a
batch dimension to the parametric gate tensors.

Background
----------
The QSVM kernel value between two data points x and y is:

    K(x, y) = |⟨0...0| φ†(y) φ(x) |0...0⟩|²

where φ(x) is a feature-map unitary that encodes x into a quantum state.
Here we use the BSP (Bravyi–Steane–Preskill) feature map from the benchmark:

    φ(x):  H on all | RZ(x[i])·RY(x[i]) per qubit | CX chain | RZ(x[i])
    φ†(y): RZ(-y[i]) | reversed CX chain | RY(-y[i])·RZ(-y[i]) | H on all

The combined kernel circuit φ†(y)·φ(x) is captured once with dummy
parameters.  For each data pair (x, y), the rotation angles are plugged in
via batch_rot_mats without re-running the CUDA-Q kernel.

Key idea
--------
Parametric gates normally have a fixed 2×2 matrix in the tensor network.
We replace those tensors with a (B, 2, 2) batch of matrices — one per data
pair — and add a free "batch index" to the einsum.  A single contraction
then returns (B,) amplitudes instead of one scalar per call.

This mirrors the technique used in benchmarks/benchmark_qsvm_v2_cudaq.py
and works directly with cuQuantum's Network API for GPU acceleration.

Usage
-----
    python examples/11_qsvm_batching.py

Optional dependencies:
    pip install torch          # for batched einsum on GPU
    pip install opt_einsum     # better CPU path optimisation
"""

import math
import time
import numpy as np
import cudaq

from cudaq_einsum import (
    capture_circuit,
    make_rotation_matrices,
    build_batched_args,
    batched_contract,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import opt_einsum  # noqa: F401
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False


# ── Feature map circuit (defined inside main to avoid slow module-level JIT) ──
# See _make_qsvm_kernel() below.


def _make_qsvm_kernel():
    """Return the QSVM combined kernel circuit (compiled once on first call)."""
    @cudaq.kernel
    def qsvm_kernel_circuit(x: list[float], neg_y: list[float], n: int):
        """
        Full QSVM kernel circuit: φ†(y)·φ(x).

        Computes K(x, y) = |⟨0...0| φ†(y) φ(x) |0...0⟩|²
        where φ(x) is the BSP feature map.

        Args:
            x:     Feature vector for the first data point (length n).
            neg_y: Negated feature vector -y for the second point (length n).
            n:     Number of qubits (== feature dimension).
        """
        q = cudaq.qvector(n)
        for i in range(n): h(q[i])
        for i in range(n):
            rz(x[i], q[i])
            ry(x[i], q[i])
        for i in range(n - 1): cx(q[i], q[i + 1])
        for i in range(n): rz(x[i], q[i])
        for i in range(n): rz(neg_y[i], q[i])
        for i in range(n - 2, -1, -1): cx(q[i], q[i + 1])
        for i in range(n):
            ry(neg_y[i], q[i])
            rz(neg_y[i], q[i])
        for i in range(n): h(q[i])

    return qsvm_kernel_circuit


# ── Helper: build angle vectors for each (x, y) pair ─────────────────────────

def build_thetas_for_pair(x, neg_y, info_list, n_gates_total):
    """
    Map data pair (x, -y) onto the rotation angles for each parametric gate,
    following the gate ordering produced by qsvm_kernel_circuit.

    Heuristic: gates in the first half of the circuit belong to φ(x),
    gates in the second half to φ†(y).  This matches the circuit structure
    above exactly.
    """
    mid = n_gates_total // 2
    thetas = np.zeros(len(info_list))
    for p_idx, inf in enumerate(info_list):
        qubit = inf.qubit
        if inf.position < mid:
            thetas[p_idx] = x[qubit]
        else:
            thetas[p_idx] = neg_y[qubit]
    return thetas


# ── Sequential baseline (for timing and correctness comparison) ───────────────

def kernel_sequential(kernel_fn, data, n):
    """
    Compute the QSVM kernel matrix one pair at a time.
    Each call to capture_circuit re-runs the CUDA-Q kernel.

    Uses opt_einsum for path optimization (required for circuits with >10 gates).

    Returns:
        kernel_matrix (n_samples, n_samples) — symmetric, diagonal = 1.
    """
    import opt_einsum

    n_samples = len(data)
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            circuit_ij = capture_circuit(
                kernel_fn, list(data[i]), list(-data[j]), n
            )
            args = circuit_ij.to_einsum_sublist_args()
            sv = opt_einsum.contract(*args).flatten()
            K[i, j] = np.abs(sv[0]) ** 2   # amplitude of |00...0⟩
            K[j, i] = K[i, j]

    return K


# ── Batched kernel (single contraction for all pairs) ─────────────────────────

def kernel_batched(kernel_fn, data, n, backend='numpy'):
    """
    Compute the QSVM kernel matrix in a single batched einsum contraction.

    Strategy:
      1. Capture the circuit once with dummy parameters.
      2. Identify parametric gates with get_parametric_gate_info().
      3. For every (i, j) pair build the rotation angle vector.
      4. Call make_rotation_matrices() to get (B, P, 2, 2) matrices.
      5. Call build_batched_args() to build the sublist einsum args.
      6. Contract once → (B,) complex amplitudes.

    Args:
        kernel_fn: The CUDA-Q kernel function.
        data:      (n_samples, n) array of data points.
        n:         Feature dimension / number of qubits.
        backend:   'numpy' or 'torch'.

    Returns:
        kernel_matrix (n_samples, n_samples).
    """
    n_samples = len(data)

    # 1. Capture once
    dummy = [0.0] * n
    circuit = capture_circuit(kernel_fn, dummy, dummy, n)

    # 2. Identify parametric gates
    positions, codes_list, info_list = circuit.get_parametric_gate_info(
        gate_names=['rx', 'ry', 'rz']
    )
    codes = np.array(codes_list)
    n_gates_total = len(circuit.gates)

    # 3. Build angle matrix for every upper-triangle pair
    pairs_i, pairs_j = np.triu_indices(n_samples, k=1)
    B = len(pairs_i)
    P = len(positions)
    thetas = np.zeros((B, P))

    for b, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        thetas[b] = build_thetas_for_pair(
            data[i], -data[j], info_list, n_gates_total
        )

    # 4. Build rotation matrices
    rot_mats = make_rotation_matrices(thetas, codes, backend=backend)

    # 5. Contract via batched_contract (uses opt_einsum when available)
    result = batched_contract(circuit, positions, rot_mats, mode='amplitude')
    if HAS_TORCH and isinstance(result, torch.Tensor):
        amplitudes = result.cpu().numpy()
    else:
        amplitudes = np.asarray(result)

    kernel_vals = np.abs(amplitudes) ** 2

    # 6. Fill symmetric matrix
    K = np.zeros((n_samples, n_samples))
    K[pairs_i, pairs_j] = kernel_vals
    K[pairs_j, pairs_i] = kernel_vals
    np.fill_diagonal(K, 1.0)   # K(x, x) = 1 by construction

    return K


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Example 11: Batched QSVM Kernel Matrix")
    print("=" * 65)

    n_features = 3          # number of qubits / feature dimensions
    n_samples  = 5          # keep small so sequential is fast enough to compare

    rng = np.random.default_rng(42)
    data = rng.uniform(-1.0, 1.0, (n_samples, n_features))

    print(f"\nData: {n_samples} samples × {n_features} features")
    print(f"Circuit: QSVM BSP feature map (φ†(y)φ(x)), {n_features} qubits")
    print(f"Pairs to evaluate: {n_samples*(n_samples-1)//2} (upper triangle)")

    # Compile the kernel once before timing
    print("\n[0] Compiling CUDA-Q kernel (one-time JIT)...", flush=True)
    kernel_fn = _make_qsvm_kernel()
    print("    Done.\n")

    # ── Sequential ────────────────────────────────────────────────────────────
    if not HAS_OPT_EINSUM:
        print("[1] Sequential — skipped (opt_einsum not installed; required for deep circuits)")
        K_seq = None
    else:
        print("[1] Sequential (one capture + opt_einsum contraction per pair)...", flush=True)
        t0 = time.perf_counter()
        K_seq = kernel_sequential(kernel_fn, data, n_features)
        t_seq = time.perf_counter() - t0
        print(f"    Done in {t_seq:.3f}s")

    # ── Batched (numpy) ───────────────────────────────────────────────────────
    print("\n[2] Batched numpy (single einsum contraction)...", flush=True)
    t0 = time.perf_counter()
    K_bat_np = kernel_batched(kernel_fn, data, n_features, backend='numpy')
    t_bat_np = time.perf_counter() - t0
    print(f"    Done in {t_bat_np:.3f}s")

    if K_seq is not None:
        match_np = np.allclose(K_seq, K_bat_np, atol=1e-4)
        print(f"    Matches sequential: {match_np}")

    # ── Batched (torch) ───────────────────────────────────────────────────────
    if HAS_TORCH:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n[3] Batched torch ({device})...", flush=True)
        t0 = time.perf_counter()
        K_bat_torch = kernel_batched(kernel_fn, data, n_features, backend='torch')
        t_bat_torch = time.perf_counter() - t0
        print(f"    Done in {t_bat_torch:.3f}s")
        if K_seq is not None:
            match_torch = np.allclose(K_seq, K_bat_torch, atol=1e-4)
            print(f"    Matches sequential: {match_torch}")
    else:
        print("\n[3] Batched torch — skipped (torch not installed)")

    # ── Display kernel matrix ────────────────────────────────────────────────
    K_display = K_seq if K_seq is not None else K_bat_np
    label = "sequential" if K_seq is not None else "batched numpy"
    print(f"\n[4] Kernel matrix ({label}):")
    print("    " + " ".join(f"  x{j}" for j in range(n_samples)))
    for i in range(n_samples):
        row = " ".join(f"{K_display[i, j]:5.3f}" for j in range(n_samples))
        print(f"    x{i} {row}")

    # ── Verify kernel properties ──────────────────────────────────────────────
    print("\n[5] Kernel matrix properties:")
    diag = np.diag(K_display)
    print(f"    Diagonal (all should be 1.0): min={diag.min():.6f}, max={diag.max():.6f}")
    print(f"    Symmetric: {np.allclose(K_display, K_display.T, atol=1e-6)}")
    eigvals = np.linalg.eigvalsh(K_display)
    print(f"    Min eigenvalue: {eigvals.min():.6f} (should be ≥ 0 for PSD)")
    print(f"    PSD: {bool(eigvals.min() >= -1e-6)}")

    # ── Show API usage ────────────────────────────────────────────────────────
    print("\n[6] API usage summary:")
    print("""
    from cudaq_einsum import (
        capture_circuit, make_rotation_matrices,
        build_batched_args, batched_contract,
    )

    # Step 1: Capture circuit once with dummy parameters
    circuit = capture_circuit(qsvm_kernel_circuit, dummy, dummy, n)

    # Step 2: Identify which gates are parametric
    positions, codes_list, info_list = circuit.get_parametric_gate_info(
        gate_names=['rx', 'ry', 'rz']
    )
    codes = np.array(codes_list)

    # Step 3: Build (B, P) angle matrix — one row per data pair
    thetas = np.zeros((B, len(positions)))
    for b, (x, neg_y) in enumerate(pairs):
        thetas[b] = assign_angles(x, neg_y, info_list, circuit)

    # Step 4: Build (B, P, 2, 2) rotation matrices
    rot_mats = make_rotation_matrices(thetas, codes, backend='torch')

    # Step 5: Single batched contraction → (B,) amplitudes
    amplitudes = batched_contract(circuit, positions, rot_mats, mode='amplitude')
    kernel_values = np.abs(amplitudes) ** 2   # (B,) real kernel values

    # For large circuits use cuQuantum Network directly:
    # args = build_batched_args(circuit, positions, rot_mats, mode='amplitude')
    # network = Network(*args, options=NetworkOptions(device_id=0))
    # amplitudes = network.contract()
""")

    print("=" * 65)


if __name__ == "__main__":
    main()
