#!/usr/bin/env python3
"""
Test script for the Einsum NVQIR simulator.

This script tests if the custom EinsumBuilder simulator can be loaded
and used with CUDA-Q kernels.
"""

import cudaq

# Set our custom simulator as the target
cudaq.set_target("einsum")

print("=== Testing Einsum Simulator ===\n")

# Define a simple GHZ state kernel
@cudaq.kernel
def ghz(n: int):
    """Create a GHZ state with n qubits."""
    qubits = cudaq.qvector(n)
    h(qubits[0])
    for i in range(1, n):
        cx(qubits[0], qubits[i])

# Define a parameterized kernel
@cudaq.kernel
def parameterized_circuit(theta: float):
    """A simple parameterized circuit."""
    q = cudaq.qvector(2)
    h(q[0])
    rx(theta, q[1])
    cx(q[0], q[1])
    ry(theta * 2, q[0])

print("--- Test 1: GHZ State (3 qubits) ---")
results = cudaq.sample(ghz, 3, shots_count=100)
print(f"Results: {results}\n")

print("--- Test 2: Parameterized Circuit (theta=0.5) ---")
results = cudaq.sample(parameterized_circuit, 0.5, shots_count=100)
print(f"Results: {results}\n")

print("--- Test 3: GHZ State (5 qubits) ---")
results = cudaq.sample(ghz, 5, shots_count=100)
print(f"Results: {results}\n")

print("=== All tests completed ===")