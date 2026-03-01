#!/usr/bin/env python3
"""
Example 01: Basic Tensor Extraction

This example demonstrates how to extract tensors and einsum expressions
from a CUDA-Q circuit for customized contraction.

The key insight: You get the raw tensors and index structure,
allowing you to use ANY tensor contraction library.
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


def main():
    print("=" * 70)
    print("Example 01: Basic Tensor Extraction")
    print("=" * 70)

    # Define a simple Bell state circuit
    @cudaq.kernel
    def bell_state():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])

    # Capture the circuit
    print("\n[1] Capturing circuit...")
    circuit = capture_circuit(bell_state)

    # Display circuit info
    print(f"\n[2] Circuit Information:")
    print(f"    Number of qubits: {circuit.num_qubits}")
    print(f"    Number of gates: {len(circuit.gates)}")
    print(f"    Max tensor index: {circuit.max_index}")
    print(f"    Initial indices: {circuit.initial_indices}")
    print(f"    Output indices: {circuit.output_indices}")

    # Show gate details
    print(f"\n[3] Gate Details:")
    for i, gate in enumerate(circuit.gates):
        print(f"    Gate {i}: {gate.name}")
        print(f"      - Controls: {gate.controls}")
        print(f"      - Targets: {gate.targets}")
        print(f"      - Input indices: {gate.input_indices}")
        print(f"      - Output indices: {gate.output_indices}")
        print(f"      - Tensor shape: {circuit.gate_tensors[i].shape}")

    # Extract tensors for custom use
    print(f"\n[4] Extracted Tensors:")
    print(f"    Initial state tensors (|0⟩ for each qubit):")
    for tensor, (idx,) in circuit.get_initial_state_tensors():
        print(f"      Index {idx}: shape={tensor.shape}, values={tensor}")

    print(f"\n    Gate tensors:")
    for i, (tensor, indices) in enumerate(zip(circuit.gate_tensors, circuit.gate_indices)):
        print(f"      Gate {i} ({circuit.gates[i].name}):")
        print(f"        Indices: {indices}")
        print(f"        Shape: {tensor.shape}")

    # Show einsum expression (subscript format)
    print(f"\n[5] Einsum Expression (subscript format):")
    subscripts, operands = circuit.to_einsum_args()
    print(f"    Subscripts: {subscripts}")
    print(f"    Number of operands: {len(operands)}")

    # Contract using numpy
    print(f"\n[6] Contraction with numpy.einsum:")
    result = np.einsum(subscripts, *operands)
    state_vector = result.flatten()
    print(f"    Result shape: {result.shape}")
    print(f"    State vector: {state_vector}")
    print(f"    Expected Bell state: [1/√2, 0, 0, 1/√2]")

    # Verify
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    if np.allclose(state_vector, expected):
        print("    ✓ Correct!")
    else:
        print("    ✗ Mismatch!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
