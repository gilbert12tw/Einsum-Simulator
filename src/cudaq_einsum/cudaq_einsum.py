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

    # Set target before running any kernel.
    cudaq.set_target("einsum")

    # Run kernel FIRST.
    #
    # We intentionally run cudaq.sample() before creating the sidecar because
    # cudaq.set_target() only registers the target — the actual plugin SO
    # (libnvqir-einsum.so) is dlopen'd by CUDA-Q the moment the first kernel
    # executes.  Only after sample() returns is the library guaranteed to be
    # present in /proc/self/maps.  We can then open the same path via ctypes,
    # obtaining the *same* dlopen instance and therefore the *same*
    # g_einsum_buffer that the C++ side wrote into.
    #
    # exportToSidecar() *overwrites* g_einsum_buffer (not appends), so no data
    # is lost by not calling clear() first.
    cudaq.sample(kernel, *args, shots_count=shots, **kwargs)

    # Now the library is in /proc/self/maps.  Force sidecar re-detection so we
    # always use the path CUDA-Q loaded (same dlopen instance).
    global _sidecar
    loaded_path = EinsumSidecar._find_already_loaded_path()
    if _sidecar is None or (loaded_path and loaded_path != _sidecar.lib_path):
        _sidecar = None
    sidecar = get_sidecar(lib_path)

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

    cudaq.set_target("einsum")
    cudaq.sample(kernel, *args, shots_count=shots, **kwargs)

    # Re-detect sidecar after sample() so we share CUDA-Q's dlopen instance.
    global _sidecar
    loaded_path = EinsumSidecar._find_already_loaded_path()
    if _sidecar is None or (loaded_path and loaded_path != _sidecar.lib_path):
        _sidecar = None
    sidecar = get_sidecar(lib_path)

    json_str = sidecar.get_circuit_json()
    if json_str is None:
        raise RuntimeError("Failed to capture circuit: buffer is empty.")

    return json_str


def reset_sidecar():
    """Reset the global sidecar instance."""
    global _sidecar
    _sidecar = None
