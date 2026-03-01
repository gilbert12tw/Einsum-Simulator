#!/usr/bin/env python3
"""
Example 03: PyTorch Tensor Contraction with Autograd

This example demonstrates using PyTorch for tensor contraction,
enabling automatic differentiation for variational quantum algorithms.

Key features:
- GPU acceleration with CUDA
- Automatic differentiation (autograd)
- Integration with PyTorch optimization
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
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Install with: pip install torch")


def main():
    print("=" * 70)
    print("Example 03: PyTorch Tensor Contraction with Autograd")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nPyTorch is required for this example.")
        print("Install with: pip install torch")
        return

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Define a parameterized circuit
    @cudaq.kernel
    def variational_circuit(theta: float):
        q = cudaq.qvector(2)
        h(q[0])
        ry(theta, q[1])
        cx(q[0], q[1])

    # Capture with a specific parameter value
    print("\n[1] Capturing variational circuit (theta=π/4)...")
    circuit = capture_circuit(variational_circuit, math.pi / 4)
    print(f"    Qubits: {circuit.num_qubits}")
    print(f"    Gates: {len(circuit.gates)}")

    # Method 1: Basic torch contraction
    print("\n[2] Method 1: Basic PyTorch contraction")
    args = circuit.to_torch_sublist_args()
    result = torch.einsum(*args)
    state_vector = result.flatten()
    print(f"    State vector: {state_vector}")

    # Method 2: GPU acceleration (if available)
    print("\n[3] Method 2: Device placement")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Using device: {device}")

    # Move tensors to device
    args_gpu = []
    for item in args:
        if isinstance(item, torch.Tensor):
            args_gpu.append(item.to(device))
        else:
            args_gpu.append(item)

    result_gpu = torch.einsum(*args_gpu)
    print(f"    Result device: {result_gpu.device}")
    print(f"    State vector: {result_gpu.flatten().cpu()}")

    # Method 3: Autograd for parameter gradients
    print("\n[4] Method 3: Automatic differentiation")
    print("    Computing gradient of <Z⊗Z> expectation with respect to RY angle")

    # For demonstration, we'll manually create tensors with gradients
    # In practice, you'd rebuild the circuit tensor with trainable parameters

    # Create the circuit tensors with gradient tracking
    theta = torch.tensor(math.pi / 4, requires_grad=True)

    # Initial states |0⟩
    zero_state = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex128)

    # Hadamard gate
    h_matrix = torch.tensor([
        [1, 1],
        [1, -1]
    ], dtype=torch.complex128) / math.sqrt(2)

    # RY gate (parameterized)
    def ry_matrix(angle):
        c = torch.cos(angle / 2)
        s = torch.sin(angle / 2)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s, c])
        ]).to(torch.complex128)

    # CNOT gate
    cnot_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex128).reshape(2, 2, 2, 2)

    # Build einsum contraction manually
    # Initial: |0⟩_a |0⟩_b
    # H on q0: H_ca @ |0⟩_a → |+⟩_c
    # RY on q1: RY_db @ |0⟩_b → |ψ⟩_d
    # CNOT: CNOT_efcd @ |+⟩_c @ |ψ⟩_d → |final⟩_ef

    ry = ry_matrix(theta)

    # Contract step by step
    psi_c = torch.einsum('ca,a->c', h_matrix, zero_state)  # H|0⟩
    psi_d = torch.einsum('db,b->d', ry, zero_state)  # RY|0⟩
    psi_ef = torch.einsum('efcd,c,d->ef', cnot_matrix, psi_c, psi_d)  # CNOT

    # Compute <Z⊗Z> expectation
    # Z⊗Z eigenvalues: |00⟩→+1, |01⟩→-1, |10⟩→-1, |11⟩→+1
    zz_eigenvalues = torch.tensor([1, -1, -1, 1], dtype=torch.complex128)
    probs = torch.abs(psi_ef.flatten()) ** 2
    expectation = torch.sum(probs * zz_eigenvalues).real

    print(f"    θ = π/4 = {theta.item():.4f}")
    print(f"    <Z⊗Z> = {expectation.item():.6f}")

    # Compute gradient
    expectation.backward()
    print(f"    d<Z⊗Z>/dθ = {theta.grad.item():.6f}")

    # Method 4: Optimization example
    print("\n[5] Method 4: Simple optimization")
    print("    Finding θ that maximizes <Z⊗Z>")

    theta_opt = torch.tensor(0.5, requires_grad=True)
    optimizer = torch.optim.Adam([theta_opt], lr=0.1)

    for step in range(20):
        optimizer.zero_grad()

        ry = ry_matrix(theta_opt)
        psi_c = torch.einsum('ca,a->c', h_matrix, zero_state)
        psi_d = torch.einsum('db,b->d', ry, zero_state)
        psi_ef = torch.einsum('efcd,c,d->ef', cnot_matrix, psi_c, psi_d)

        probs = torch.abs(psi_ef.flatten()) ** 2
        expectation = torch.sum(probs * zz_eigenvalues).real

        # Maximize by minimizing negative
        loss = -expectation
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 19:
            print(f"    Step {step:2d}: θ={theta_opt.item():.4f}, <Z⊗Z>={expectation.item():.6f}")

    print(f"\n    Final θ = {theta_opt.item():.4f} (optimal is 0 or π)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
