"""
Einsum NVQIR Python Integration

This package provides Python bindings and utilities for capturing
quantum circuits from CUDA-Q kernels and converting them to tensor
network (einsum) representations.

Modules:
    einsum_sidecar: ctypes client for C++ sidecar API
    einsum_circuit: EinsumCircuit data class for circuit representation
    cudaq_einsum: High-level API for circuit capture
"""

__version__ = "0.1.0"

# Support both package import and direct path import
try:
    from .einsum_sidecar import EinsumSidecar
    from .einsum_circuit import EinsumCircuit
    from .cudaq_einsum import capture_circuit
except ImportError:
    from einsum_sidecar import EinsumSidecar
    from einsum_circuit import EinsumCircuit
    from cudaq_einsum import capture_circuit

__all__ = ["EinsumSidecar", "EinsumCircuit", "capture_circuit"]
