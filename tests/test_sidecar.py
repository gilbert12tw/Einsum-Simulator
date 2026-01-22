#!/usr/bin/env python3
"""
Test script for the Sidecar ctypes approach.
This tests whether we can extract circuit data from the C++ simulator via ctypes.
"""

import cudaq
import ctypes
import json
import os

# Path to the einsum library
LIB_PATH = "/opt/nvidia/cudaq/lib/libnvqir-einsum.so"

def load_einsum_library():
    """Load the einsum library and setup ctypes interfaces."""
    if not os.path.exists(LIB_PATH):
        raise FileNotFoundError(f"Library not found: {LIB_PATH}")

    lib = ctypes.CDLL(LIB_PATH)

    # Setup function signatures
    lib.get_einsum_length.argtypes = []
    lib.get_einsum_length.restype = ctypes.c_int

    lib.get_einsum_data.argtypes = [ctypes.c_char_p]
    lib.get_einsum_data.restype = None

    lib.clear_einsum_buffer.argtypes = []
    lib.clear_einsum_buffer.restype = None

    return lib

def get_einsum_json(lib):
    """Retrieve the einsum JSON from the sidecar buffer."""
    length = lib.get_einsum_length()
    if length <= 0:
        return None

    # Create buffer and copy data
    buffer = ctypes.create_string_buffer(length + 1)
    lib.get_einsum_data(buffer)

    # Decode and parse JSON
    json_str = buffer.value.decode('utf-8')
    return json_str

def main():
    print("=" * 60)
    print("Testing Sidecar Ctypes Approach")
    print("=" * 60)

    # Step 1: Load library
    print("\n[1] Loading einsum library...")
    try:
        lib = load_einsum_library()
        print(f"    Library loaded: {LIB_PATH}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

    # Step 2: Set target to einsum
    print("\n[2] Setting CUDA-Q target to 'einsum'...")
    cudaq.set_target("einsum")
    print("    Target set successfully")

    # Step 3: Define and run a simple kernel
    print("\n[3] Defining test kernel (GHZ state)...")

    @cudaq.kernel
    def ghz_kernel():
        q = cudaq.qvector(3)
        h(q[0])
        cx(q[0], q[1])
        cx(q[1], q[2])

    print("    Kernel defined")

    # Step 4: Clear buffer and run kernel
    print("\n[4] Clearing buffer and running kernel...")
    lib.clear_einsum_buffer()

    result = cudaq.sample(ghz_kernel, shots_count=10)
    print(f"    Sample result: {result}")

    # Step 5: Retrieve sidecar data
    print("\n[5] Retrieving data from sidecar buffer...")
    json_str = get_einsum_json(lib)

    if json_str:
        print(f"    Buffer length: {len(json_str)} bytes")
        print("\n" + "-" * 60)
        print("RAW JSON OUTPUT:")
        print("-" * 60)
        print(json_str)

        # Parse and display structured data
        print("-" * 60)
        print("PARSED DATA:")
        print("-" * 60)
        try:
            data = json.loads(json_str)
            print(f"  Number of qubits: {data['numQubits']}")
            print(f"  Number of gates: {data['numGates']}")
            print(f"  Max index used: {data['maxIndex']}")
            print(f"  Output indices: {data['outputIndices']}")
            print()
            print("  Gates:")
            for i, gate in enumerate(data['gates']):
                print(f"    [{i}] {gate['name']}: in={gate['inputIndices']} -> out={gate['outputIndices']}")
                if gate['matrix']:
                    print(f"        matrix size: {len(gate['matrix'])} elements")
            print()
            print("SUCCESS! Sidecar approach works!")
            return True
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
            return False
    else:
        print("    ERROR: Buffer is empty!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
