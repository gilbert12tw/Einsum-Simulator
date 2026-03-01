#!/usr/bin/env python3
"""
Example 05: VQE Ansatz Tensor Extraction

This example demonstrates extracting tensor networks from VQE-style
variational ansatz circuits, showing how the tensor structure can be
reused while only updating parameter-dependent tensors.

Key features:
- Hardware-efficient ansatz (RY-CNOT layers)
- Tensor structure extraction for custom optimization
- Separating fixed vs. parameterized tensors
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
    print("Example 05: VQE Ansatz Tensor Extraction")
    print("=" * 70)

    # Define a hardware-efficient ansatz
    @cudaq.kernel
    def hea_ansatz(theta0: float, theta1: float, theta2: float, theta3: float):
        """
        Hardware-Efficient Ansatz (2-qubit, 1 layer)

        Layer structure:
        q0: ─RY(θ0)─●─RY(θ2)─
                   │
        q1: ─RY(θ1)─X─RY(θ3)─
        """
        q = cudaq.qvector(2)

        # First RY layer
        ry(theta0, q[0])
        ry(theta1, q[1])

        # Entangling layer
        cx(q[0], q[1])

        # Second RY layer
        ry(theta2, q[0])
        ry(theta3, q[1])

    # Capture with sample parameters
    print("\n[1] Capturing HEA ansatz with sample parameters...")
    params = (0.1, 0.2, 0.3, 0.4)
    circuit = capture_circuit(hea_ansatz, *params)

    print(f"    Parameters: θ = {params}")
    print(f"    Qubits: {circuit.num_qubits}")
    print(f"    Gates: {len(circuit.gates)}")

    # Analyze circuit structure
    print("\n[2] Circuit structure analysis:")
    for i, gate in enumerate(circuit.gates):
        param_str = f"({gate.params[0]:.2f})" if gate.params else ""
        print(f"    Gate {i}: {gate.name}{param_str}")
        print(f"           indices: {gate.input_indices} → {gate.output_indices}")
        print(f"           tensor shape: {circuit.gate_tensors[i].shape}")

    # Show how structure stays fixed while tensors change
    print("\n[3] Demonstrating parameter updates:")
    print("    The einsum structure (index pattern) stays fixed,")
    print("    only the RY gate tensors change with parameters.")

    subscripts, _ = circuit.to_einsum_args()
    print(f"\n    Fixed subscripts: {subscripts}")

    # Compute state for different parameters
    print("\n    Computing states for different parameters:")
    param_sets = [
        (0.0, 0.0, 0.0, 0.0),
        (math.pi/4, math.pi/4, 0.0, 0.0),
        (math.pi/2, math.pi/2, math.pi/2, math.pi/2),
    ]

    for params in param_sets:
        circuit = capture_circuit(hea_ansatz, *params)
        state = circuit.get_state_vector()
        print(f"    θ={[f'{p:.2f}' for p in params]}")
        print(f"      |ψ⟩ = {state}")

    # Compute expectation value <Z⊗Z>
    print("\n[4] Computing <Z⊗Z> expectation values:")

    def compute_zz_expectation(circuit):
        """Compute <ψ|Z⊗Z|ψ> for a 2-qubit state."""
        state = circuit.get_state_vector()
        probs = np.abs(state) ** 2
        # Z⊗Z eigenvalues: |00⟩→+1, |01⟩→-1, |10⟩→-1, |11⟩→+1
        zz_eigenvalues = np.array([1, -1, -1, 1])
        return np.sum(probs * zz_eigenvalues)

    print("\n    θ values                    | <Z⊗Z>")
    print("    " + "-" * 50)
    for params in param_sets:
        circuit = capture_circuit(hea_ansatz, *params)
        exp_val = compute_zz_expectation(circuit)
        print(f"    {[f'{p:.2f}' for p in params]:30s} | {exp_val:.6f}")

    # VQE-style optimization scan
    print("\n[5] Parameter scan (1D slice):")
    print("    Scanning θ₀ while keeping θ₁=θ₂=θ₃=0")

    scan_values = np.linspace(0, 2 * math.pi, 21)
    energies = []

    for theta0 in scan_values:
        circuit = capture_circuit(hea_ansatz, theta0, 0.0, 0.0, 0.0)
        energy = compute_zz_expectation(circuit)
        energies.append(energy)

    # Find minimum
    min_idx = np.argmin(energies)
    print(f"\n    Minimum at θ₀ = {scan_values[min_idx]:.4f}")
    print(f"    Min <Z⊗Z> = {energies[min_idx]:.6f}")

    # ASCII plot
    print("\n    Energy landscape:")
    min_e, max_e = min(energies), max(energies)
    for i, (theta, e) in enumerate(zip(scan_values[::4], energies[::4])):
        bar_len = int((e - min_e) / (max_e - min_e + 1e-10) * 30) + 1
        print(f"    θ={theta:.2f}: {'█' * bar_len} {e:.3f}")

    # Show tensor reuse pattern with opt_einsum
    if HAS_OPT_EINSUM:
        print("\n[6] Contraction optimization with opt_einsum:")

        # Get structure once
        circuit = capture_circuit(hea_ansatz, 0.0, 0.0, 0.0, 0.0)
        subscripts, operands = circuit.to_einsum_args()

        # Create reusable expression
        shapes = [op.shape for op in operands]
        expr = oe.contract_expression(subscripts, *shapes, optimize='optimal')

        print(f"    Created reusable contraction expression")
        print(f"    Operand shapes: {shapes}")

        # Path info
        path, info = oe.contract_path(subscripts, *operands, optimize='optimal')
        print(f"    Contraction path: {path}")
        print(f"    FLOP count per evaluation: {info.opt_cost:.2e}")

        # Demonstrate reuse
        print("\n    Timing comparison (conceptual):")
        print("    - Without reuse: path optimization + contraction each time")
        print("    - With reuse: contraction only (path cached)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
