"""
High-level API for capturing CUDA-Q circuits as tensor networks.

This module provides the main user-facing function for circuit capture,
integrating the sidecar client with circuit construction.
"""

from typing import Any, Optional, Callable

# Support both package import and direct path import
try:
    from .einsum_sidecar import EinsumSidecar
    from .einsum_circuit import EinsumCircuit
except ImportError:
    from einsum_sidecar import EinsumSidecar
    from einsum_circuit import EinsumCircuit


# Global sidecar instance (lazily initialized)
_sidecar: Optional[EinsumSidecar] = None


def get_sidecar(lib_path: Optional[str] = None) -> EinsumSidecar:
    """
    Get or create the global sidecar instance.

    Args:
        lib_path: Optional path to libnvqir-einsum.so.

    Returns:
        EinsumSidecar instance.
    """
    global _sidecar
    if _sidecar is None:
        _sidecar = EinsumSidecar(lib_path)
    return _sidecar


def capture_circuit(
    kernel: Callable,
    *args,
    lib_path: Optional[str] = None,
    shots: int = 1,
    **kwargs
) -> EinsumCircuit:
    """
    Execute a CUDA-Q kernel and capture the circuit structure.

    This function:
    1. Sets the CUDA-Q target to 'einsum'
    2. Clears the sidecar buffer
    3. Runs the kernel via cudaq.sample()
    4. Retrieves the circuit JSON from the sidecar
    5. Builds and returns an EinsumCircuit

    Args:
        kernel: CUDA-Q kernel function decorated with @cudaq.kernel.
        *args: Arguments to pass to the kernel.
        lib_path: Optional path to libnvqir-einsum.so.
        shots: Number of shots for cudaq.sample() (default 1).
        **kwargs: Additional keyword arguments for the kernel.

    Returns:
        EinsumCircuit containing the captured circuit structure.

    Raises:
        RuntimeError: If circuit capture fails.
        ImportError: If cudaq is not available.

    Example:
        import cudaq
        from python.cudaq_einsum import capture_circuit

        @cudaq.kernel
        def ghz_kernel(n: int):
            q = cudaq.qvector(n)
            h(q[0])
            for i in range(1, n):
                cx(q[0], q[i])

        circuit = capture_circuit(ghz_kernel, 3)
        state = circuit.get_state_vector()
    """
    try:
        import cudaq
    except ImportError:
        raise ImportError(
            "cudaq is required. Please run in a CUDA-Q environment."
        )

    # Get sidecar
    sidecar = get_sidecar(lib_path)

    # Set target to einsum
    cudaq.set_target("einsum")

    # Clear buffer
    sidecar.clear()

    # Run kernel - use sample() to trigger circuit execution
    cudaq.sample(kernel, *args, shots_count=shots, **kwargs)

    # Retrieve JSON
    json_str = sidecar.get_circuit_json()
    if json_str is None:
        raise RuntimeError(
            "Failed to capture circuit: sidecar buffer is empty. "
            "Ensure the kernel executes quantum operations."
        )

    # Build and return circuit
    return EinsumCircuit.from_json(json_str)


def capture_circuit_json(
    kernel: Callable,
    *args,
    lib_path: Optional[str] = None,
    shots: int = 1,
    **kwargs
) -> str:
    """
    Execute a CUDA-Q kernel and return the raw circuit JSON.

    This is a lower-level function useful for debugging or when you
    need the raw JSON output.

    Args:
        kernel: CUDA-Q kernel function.
        *args: Arguments to pass to the kernel.
        lib_path: Optional path to libnvqir-einsum.so.
        shots: Number of shots for cudaq.sample().
        **kwargs: Additional keyword arguments.

    Returns:
        JSON string containing the circuit data.

    Raises:
        RuntimeError: If capture fails.
    """
    try:
        import cudaq
    except ImportError:
        raise ImportError("cudaq is required.")

    sidecar = get_sidecar(lib_path)
    cudaq.set_target("einsum")
    sidecar.clear()
    cudaq.sample(kernel, *args, shots_count=shots, **kwargs)

    json_str = sidecar.get_circuit_json()
    if json_str is None:
        raise RuntimeError("Failed to capture circuit: buffer is empty.")

    return json_str


def reset_sidecar():
    """Reset the global sidecar instance."""
    global _sidecar
    _sidecar = None
