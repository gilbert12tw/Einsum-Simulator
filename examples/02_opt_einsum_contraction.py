#!/usr/bin/env python3
"""
Example 02: Optimized Contraction with opt_einsum

This example demonstrates using opt_einsum for optimized tensor contraction.
opt_einsum finds the optimal contraction order, which can dramatically
speed up computation for larger circuits.

Key features:
- Automatic contraction path optimization
- Support for different optimization strategies
- Path reuse for repeated contractions (e.g., VQE)
"""

import sys
import os

# Add python directory to path
_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')
sys.path.insert(0, _python_dir)

import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
from einsum_circuit import EinsumCircuit

try:
    import opt_einsum as oe
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False
    print("Warning: opt_einsum not installed. Install with: pip install opt-einsum")


def main():
    print("=" * 70)
    print("Example 02: Optimized Contraction with opt_einsum")
    print("=" * 70)

    if not HAS_OPT_EINSUM:
        print("\nopt_einsum is required for this example.")
        print("Install with: pip install opt-einsum")
        return

    # Define a larger circuit (5-qubit GHZ)
    @cudaq.kernel
    def ghz5():
        q = cudaq.qvector(5)
        h(q[0])
        for i in range(1, 5):
            cx(q[0], q[i])

    # Capture the circuit
    print("\n[1] Capturing 5-qubit GHZ circuit...")
    circuit = capture_circuit(ghz5)
    print(f"    Qubits: {circuit.num_qubits}")
    print(f"    Gates: {len(circuit.gates)}")

    # Get einsum arguments
    subscripts, operands = circuit.to_einsum_args()
    print(f"\n[2] Einsum expression: {subscripts}")

    # Method 1: Direct contraction with opt_einsum
    print("\n[3] Method 1: Direct opt_einsum contraction")
    result = oe.contract(subscripts, *operands)
    state_vector = result.flatten()
    print(f"    State vector (first 4 and last 4 amplitudes):")
    print(f"    {state_vector[:4]} ... {state_vector[-4:]}")

    # Verify GHZ state: (|00000⟩ + |11111⟩) / √2
    expected = np.zeros(32, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[31] = 1 / np.sqrt(2)
    print(f"    ✓ Correct!" if np.allclose(state_vector, expected) else "    ✗ Mismatch!")

    # Method 2: Find and display optimal contraction path
    print("\n[4] Method 2: Analyze contraction path")
    path_info = oe.contract_path(subscripts, *operands)
    print(f"    Contraction path: {path_info[0]}")
    print(f"    Optimization info:")
    print(f"      - Naive scaling: {path_info[1].naive_cost}")
    print(f"      - Optimized scaling: {path_info[1].opt_cost}")
    print(f"      - Speedup: {path_info[1].naive_cost / path_info[1].opt_cost:.2f}x")
    print(f"      - Largest intermediate: {path_info[1].largest_intermediate}")

    # Method 3: Reusable contraction expression (for VQE-like loops)
    print("\n[5] Method 3: Reusable contraction expression")
    print("    (Useful for VQE where circuit structure is fixed)")

    # Create a reusable expression
    expr = oe.contract_expression(
        subscripts,
        *[op.shape for op in operands],
        optimize='optimal'
    )
    print(f"    Created reusable expression")

    # Simulate "multiple evaluations" (in VQE, tensors would change with parameters)
    print("    Evaluating 3 times with same expression...")
    for i in range(3):
        result = expr(*operands)
        print(f"      Evaluation {i+1}: max amplitude = {np.abs(result).max():.6f}")

    # Method 4: Different optimization strategies
    print("\n[6] Method 4: Optimization strategies comparison")
    strategies = ['auto', 'greedy', 'optimal']
    for strategy in strategies:
        path, info = oe.contract_path(subscripts, *operands, optimize=strategy)
        print(f"    {strategy:8s}: cost={info.opt_cost:.2e}, largest={info.largest_intermediate}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
