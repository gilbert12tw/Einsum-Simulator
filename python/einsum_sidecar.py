"""
Sidecar client for retrieving circuit data from the C++ einsum simulator.

This module provides a ctypes-based interface to the extern "C" functions
exported by libnvqir-einsum.so for inter-process data transfer.
"""

import ctypes
import os
from typing import Optional


class EinsumSidecar:
    """
    Client for the C++ einsum simulator sidecar API.

    The sidecar provides thread-local storage that can be accessed via ctypes
    to retrieve circuit data captured by the EinsumBuilder simulator.

    Usage:
        sidecar = EinsumSidecar()
        sidecar.clear()
        # ... run CUDA-Q kernel with einsum target ...
        json_str = sidecar.get_circuit_json()
    """

    # Default library paths to search
    DEFAULT_PATHS = [
        "/opt/nvidia/cudaq/lib/libnvqir-einsum.so",
        "./build/libnvqir-einsum.so",
        "../build/libnvqir-einsum.so",
    ]

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the sidecar client.

        Args:
            lib_path: Path to libnvqir-einsum.so. If None, searches default paths.

        Raises:
            FileNotFoundError: If the library cannot be found.
            OSError: If the library fails to load.
        """
        self._lib_path = self._find_library(lib_path)
        self._lib = self._load_library()
        self._setup_ctypes()

    def _find_library(self, lib_path: Optional[str]) -> str:
        """Find the einsum library."""
        if lib_path is not None:
            if os.path.exists(lib_path):
                return lib_path
            raise FileNotFoundError(f"Library not found: {lib_path}")

        # Search default paths
        for path in self.DEFAULT_PATHS:
            if os.path.exists(path):
                return path

        # Try environment variable
        env_path = os.environ.get("EINSUM_LIB_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        raise FileNotFoundError(
            f"libnvqir-einsum.so not found. Searched: {self.DEFAULT_PATHS}. "
            "Set EINSUM_LIB_PATH environment variable or pass lib_path explicitly."
        )

    def _load_library(self) -> ctypes.CDLL:
        """Load the shared library."""
        return ctypes.CDLL(self._lib_path)

    def _setup_ctypes(self):
        """Setup ctypes function signatures."""
        # int get_einsum_length()
        self._lib.get_einsum_length.argtypes = []
        self._lib.get_einsum_length.restype = ctypes.c_int

        # void get_einsum_data(char* buffer)
        self._lib.get_einsum_data.argtypes = [ctypes.c_char_p]
        self._lib.get_einsum_data.restype = None

        # void clear_einsum_buffer()
        self._lib.clear_einsum_buffer.argtypes = []
        self._lib.clear_einsum_buffer.restype = None

    @property
    def lib_path(self) -> str:
        """Return the path to the loaded library."""
        return self._lib_path

    def get_buffer_length(self) -> int:
        """Get the current buffer length in bytes."""
        return self._lib.get_einsum_length()

    def get_circuit_json(self) -> Optional[str]:
        """
        Retrieve the circuit JSON from the sidecar buffer.

        Returns:
            JSON string containing circuit data, or None if buffer is empty.
        """
        length = self._lib.get_einsum_length()
        if length <= 0:
            return None

        # Create buffer and copy data
        buffer = ctypes.create_string_buffer(length + 1)
        self._lib.get_einsum_data(buffer)

        # Decode to string
        return buffer.value.decode('utf-8')

    def clear(self):
        """Clear the sidecar buffer."""
        self._lib.clear_einsum_buffer()

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._lib.get_einsum_length() <= 0
