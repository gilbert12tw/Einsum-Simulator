#!/usr/bin/env python3
"""
Example 04: Quantum Support Vector Machine (QSVM) Kernel

This example demonstrates extracting tensor networks from QSVM-style
feature map circuits for computing quantum kernel matrices.

QSVM uses quantum feature maps to embed classical data into quantum states,
then computes kernel values as |⟨ψ(x)|ψ(y)⟩|².

Key features:
- ZZFeatureMap-style circuit
- Kernel matrix computation via tensor contraction
- Batch processing for efficiency
"""

import sys
import os

# Add python directory to path
_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')
sys.path.insert(0, _python_dir)

import math
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
from einsum_circuit import EinsumCircuit

try:
    import opt_einsum as oe
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False


def main():
    print("=" * 70)
    print("Example 04: QSVM Kernel Computation")
    print("=" * 70)

    # Define a ZZ Feature Map circuit (simplified version)
    # This encodes 2D classical data into a 2-qubit quantum state
    @cudaq.kernel
    def zz_feature_map(x0: float, x1: float):
        """
        ZZ Feature Map for 2D data.
        Similar to Qiskit's ZZFeatureMap with 1 repetition.
        """
        q = cudaq.qvector(2)

        # First layer: Hadamard + rotation
        h(q[0])
        h(q[1])
        rz(2 * x0, q[0])
        rz(2 * x1, q[1])

        # Entangling layer: ZZ interaction
        cx(q[0], q[1])
        rz(2 * (math.pi - x0) * (math.pi - x1), q[1])
        cx(q[0], q[1])

    # Sample data points
    print("\n[1] Sample 2D data points:")
    data_points = [
        (0.1, 0.2),
        (0.3, 0.4),
        (0.5, 0.1),
        (0.2, 0.6),
    ]
    for i, (x0, x1) in enumerate(data_points):
        print(f"    Point {i}: ({x0}, {x1})")

    # Capture circuits for each data point
    print("\n[2] Capturing feature map circuits...")
    circuits = []
    for x0, x1 in data_points:
        circuit = capture_circuit(zz_feature_map, x0, x1)
        circuits.append(circuit)
        print(f"    Point ({x0}, {x1}): {len(circuit.gates)} gates")

    # Compute kernel matrix
    print("\n[3] Computing quantum kernel matrix...")
    print("    K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²")

    n = len(data_points)
    kernel_matrix = np.zeros((n, n))

    for i in range(n):
        state_i = circuits[i].get_state_vector()
        for j in range(n):
            state_j = circuits[j].get_state_vector()
            # Kernel value is |⟨ψᵢ|ψⱼ⟩|²
            inner_product = np.vdot(state_i, state_j)
            kernel_matrix[i, j] = np.abs(inner_product) ** 2

    print("\n    Kernel Matrix:")
    print("    " + "-" * 40)
    for i in range(n):
        row = " ".join(f"{kernel_matrix[i, j]:.4f}" for j in range(n))
        print(f"    | {row} |")
    print("    " + "-" * 40)

    # Verify properties
    print("\n[4] Kernel matrix properties:")
    print(f"    Diagonal (should be 1.0): {np.diag(kernel_matrix)}")
    print(f"    Symmetric: {np.allclose(kernel_matrix, kernel_matrix.T)}")
    print(f"    Positive semi-definite: {np.all(np.linalg.eigvalsh(kernel_matrix) >= -1e-10)}")

    # Show how to use extracted tensors directly
    print("\n[5] Direct tensor manipulation example:")
    print("    (For custom kernel computation pipelines)")

    circuit = circuits[0]
    print(f"\n    Circuit structure for point (0.1, 0.2):")
    print(f"    - Number of tensors: {len(circuit.gate_tensors) + circuit.num_qubits}")
    print(f"    - Index labels used: 0 to {circuit.max_index - 1}")

    # Show einsum expression
    subscripts, operands = circuit.to_einsum_args()
    print(f"\n    Einsum subscripts: {subscripts}")

    # Using opt_einsum for path analysis
    if HAS_OPT_EINSUM:
        print("\n[6] Contraction path analysis with opt_einsum:")
        path, info = oe.contract_path(subscripts, *operands, optimize='optimal')
        print(f"    Optimal contraction path: {path}")
        print(f"    FLOP count: {info.opt_cost:.2e}")
        print(f"    Memory peak: {info.largest_intermediate} elements")

    # Demonstrate batch kernel computation
    print("\n[7] Batch kernel computation pattern:")
    print("    (Efficient for large datasets)")

    # Pre-compute all state vectors
    states = [circuit.get_state_vector() for circuit in circuits]

    # Vectorized kernel computation
    states_matrix = np.array(states)  # Shape: (n, 2^num_qubits)
    gram_matrix = states_matrix @ states_matrix.conj().T  # ⟨ψᵢ|ψⱼ⟩
    kernel_matrix_fast = np.abs(gram_matrix) ** 2

    print(f"    States matrix shape: {states_matrix.shape}")
    print(f"    Kernel matches: {np.allclose(kernel_matrix, kernel_matrix_fast)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
