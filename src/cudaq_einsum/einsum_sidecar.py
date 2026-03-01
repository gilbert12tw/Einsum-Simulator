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

    # Default library paths to search (in priority order)
    DEFAULT_PATHS = [
        # 1. Bundled inside the installed Python package (pip install .)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnvqir-einsum.so"),
        # 2. CUDA-Q system installation (Docker / manual install)
        "/opt/nvidia/cudaq/lib/libnvqir-einsum.so",
        # 3. Local build directory (development)
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

    @staticmethod
    def _find_already_loaded_path() -> Optional[str]:
        """
        Search /proc/self/maps for libnvqir-einsum.so that is already mapped
        into the process by CUDA-Q.  Returns the canonical path, or None.

        This is the key to getting the SAME dlopen handle as CUDA-Q: Linux
        dlopen() deduplicates by canonical path string, so we must pass the
        exact path that CUDA-Q used — not a copy at a different location.
        """
        try:
            with open("/proc/self/maps", "r") as f:
                for line in f:
                    if "libnvqir-einsum.so" in line:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            path = parts[-1]
                            if os.path.exists(path):
                                return path
        except Exception:
            pass
        return None

    def _find_library(self, lib_path: Optional[str]) -> str:
        """Find the einsum library, preferring the instance already loaded by CUDA-Q."""
        if lib_path is not None:
            if os.path.exists(lib_path):
                return lib_path
            raise FileNotFoundError(f"Library not found: {lib_path}")

        # 1. EINSUM_LIB_PATH env var (explicit override)
        env_path = os.environ.get("EINSUM_LIB_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # 2. Already loaded by CUDA-Q — MOST IMPORTANT: same dlopen instance
        #    guarantees shared global g_einsum_buffer
        loaded = self._find_already_loaded_path()
        if loaded:
            return loaded

        # 3. Static search paths
        search_paths = list(self.DEFAULT_PATHS)
        cudaq_root = os.environ.get("CUDAQ_ROOT")
        if cudaq_root:
            search_paths.insert(0, os.path.join(cudaq_root, "lib", "libnvqir-einsum.so"))

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "libnvqir-einsum.so not found.\n"
            f"Searched: {search_paths}\n"
            "Options:\n"
            "  - pip install . then run: python -c \"import cudaq_einsum; cudaq_einsum.install_cudaq_target()\"\n"
            "  - Run: ./scripts/local-build.sh\n"
            "  - Set EINSUM_LIB_PATH=/path/to/libnvqir-einsum.so\n"
            "  - Pass lib_path explicitly to EinsumSidecar()"
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
