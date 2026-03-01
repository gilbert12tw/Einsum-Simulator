#!/usr/bin/env python3
"""
Example 06: Custom Contraction Backends

This example demonstrates how to use the extracted tensor network
with different contraction backends, allowing users to choose the
best tool for their specific use case.

Supported backends demonstrated:
- NumPy (baseline)
- opt_einsum (optimized paths)
- PyTorch (GPU, autograd)
- JAX (JIT, GPU, autograd) [if available]
"""

import sys
import os

# Add python directory to path
_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')
sys.path.insert(0, _python_dir)

import math
import time
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
from einsum_circuit import EinsumCircuit

# Check available backends
BACKENDS = {'numpy': True}

try:
    import opt_einsum as oe
    BACKENDS['opt_einsum'] = True
except ImportError:
    BACKENDS['opt_einsum'] = False

try:
    import torch
    BACKENDS['torch'] = True
except ImportError:
    BACKENDS['torch'] = False

try:
    import jax
    import jax.numpy as jnp
    BACKENDS['jax'] = True
except ImportError:
    BACKENDS['jax'] = False


def contract_numpy(circuit):
    """Contract using numpy.einsum"""
    subscripts, operands = circuit.to_einsum_args()
    return np.einsum(subscripts, *operands)


def contract_numpy_sublist(circuit):
    """Contract using numpy.einsum with sublist format"""
    args = circuit.to_einsum_sublist_args()
    return np.einsum(*args)


def contract_opt_einsum(circuit, optimize='optimal'):
    """Contract using opt_einsum with path optimization"""
    subscripts, operands = circuit.to_einsum_args()
    return oe.contract(subscripts, *operands, optimize=optimize)


def contract_torch(circuit, device='cpu'):
    """Contract using PyTorch"""
    args = circuit.to_torch_sublist_args()
    # Move to device
    args_device = []
    for item in args:
        if isinstance(item, torch.Tensor):
            args_device.append(item.to(device))
        else:
            args_device.append(item)
    return torch.einsum(*args_device)


def contract_jax(circuit):
    """Contract using JAX"""
    args = circuit.to_einsum_sublist_args()
    # Convert to JAX arrays
    jax_args = []
    for item in args:
        if isinstance(item, np.ndarray):
            jax_args.append(jnp.array(item))
        else:
            jax_args.append(item)
    return jnp.einsum(*jax_args)


def main():
    print("=" * 70)
    print("Example 06: Custom Contraction Backends")
    print("=" * 70)

    # Show available backends
    print("\n[1] Available backends:")
    for name, available in BACKENDS.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"    {name:12s}: {status}")

    # Create test circuit (4-qubit GHZ)
    @cudaq.kernel
    def ghz4():
        q = cudaq.qvector(4)
        h(q[0])
        cx(q[0], q[1])
        cx(q[1], q[2])
        cx(q[2], q[3])

    print("\n[2] Capturing 4-qubit GHZ circuit...")
    circuit = capture_circuit(ghz4)
    print(f"    Qubits: {circuit.num_qubits}")
    print(f"    Gates: {len(circuit.gates)}")

    # Expected result: (|0000⟩ + |1111⟩) / √2
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[15] = 1 / np.sqrt(2)

    # Run each backend
    print("\n[3] Running contractions with each backend:")
    print("    " + "-" * 55)

    results = {}

    # NumPy subscript format
    print("\n    numpy (subscript):")
    result = contract_numpy(circuit)
    state = result.flatten()
    correct = np.allclose(state, expected)
    print(f"      Shape: {result.shape}")
    print(f"      Correct: {correct}")
    results['numpy_subscript'] = state

    # NumPy sublist format
    print("\n    numpy (sublist):")
    result = contract_numpy_sublist(circuit)
    state = result.flatten()
    correct = np.allclose(state, expected)
    print(f"      Shape: {result.shape}")
    print(f"      Correct: {correct}")
    results['numpy_sublist'] = state

    # opt_einsum
    if BACKENDS['opt_einsum']:
        print("\n    opt_einsum:")
        result = contract_opt_einsum(circuit)
        state = result.flatten()
        correct = np.allclose(state, expected)
        print(f"      Shape: {result.shape}")
        print(f"      Correct: {correct}")
        results['opt_einsum'] = state

        # Show path info
        subscripts, operands = circuit.to_einsum_args()
        _, info = oe.contract_path(subscripts, *operands, optimize='optimal')
        print(f"      FLOP count: {info.opt_cost:.2e}")

    # PyTorch
    if BACKENDS['torch']:
        print("\n    torch (CPU):")
        result = contract_torch(circuit, device='cpu')
        state = result.flatten().numpy()
        correct = np.allclose(state, expected)
        print(f"      Shape: {tuple(result.shape)}")
        print(f"      Correct: {correct}")
        results['torch_cpu'] = state

        if torch.cuda.is_available():
            print("\n    torch (CUDA):")
            result = contract_torch(circuit, device='cuda')
            state = result.flatten().cpu().numpy()
            correct = np.allclose(state, expected)
            print(f"      Shape: {tuple(result.shape)}")
            print(f"      Device: {result.device}")
            print(f"      Correct: {correct}")
            results['torch_cuda'] = state

    # JAX
    if BACKENDS['jax']:
        print("\n    jax:")
        result = contract_jax(circuit)
        state = np.array(result.flatten())
        correct = np.allclose(state, expected)
        print(f"      Shape: {result.shape}")
        print(f"      Correct: {correct}")
        results['jax'] = state

    # Verify all results match
    print("\n[4] Cross-backend verification:")
    reference = results['numpy_subscript']
    all_match = True
    for name, state in results.items():
        match = np.allclose(state, reference)
        all_match = all_match and match
        status = "✓" if match else "✗"
        print(f"    {name:20s}: {status}")
    print(f"\n    All backends agree: {'Yes' if all_match else 'No'}")

    # Timing comparison (for larger circuit)
    print("\n[5] Timing comparison (10-qubit GHZ, 100 iterations):")

    @cudaq.kernel
    def ghz10():
        q = cudaq.qvector(10)
        h(q[0])
        for i in range(1, 10):
            cx(q[0], q[i])

    circuit10 = capture_circuit(ghz10)
    print(f"    Circuit: {circuit10.num_qubits} qubits, {len(circuit10.gates)} gates")

    n_iter = 100

    # Time numpy
    start = time.time()
    for _ in range(n_iter):
        _ = contract_numpy_sublist(circuit10)
    numpy_time = time.time() - start
    print(f"\n    numpy:      {numpy_time:.4f}s ({numpy_time/n_iter*1000:.2f}ms/iter)")

    # Time opt_einsum
    if BACKENDS['opt_einsum']:
        # With path caching
        subscripts, operands = circuit10.to_einsum_args()
        expr = oe.contract_expression(subscripts, *[op.shape for op in operands], optimize='optimal')

        start = time.time()
        for _ in range(n_iter):
            _ = expr(*operands)
        oe_time = time.time() - start
        print(f"    opt_einsum: {oe_time:.4f}s ({oe_time/n_iter*1000:.2f}ms/iter)")
        print(f"    Speedup:    {numpy_time/oe_time:.2f}x")

    # Time torch
    if BACKENDS['torch']:
        args = circuit10.to_torch_sublist_args()

        start = time.time()
        for _ in range(n_iter):
            _ = torch.einsum(*args)
        torch_time = time.time() - start
        print(f"    torch:      {torch_time:.4f}s ({torch_time/n_iter*1000:.2f}ms/iter)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
