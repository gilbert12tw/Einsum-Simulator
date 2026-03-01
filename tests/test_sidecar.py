#!/usr/bin/env python3
"""
Test script for the Sidecar ctypes approach.
This tests whether we can extract circuit data from the C++ simulator via ctypes.
"""

import cudaq
import json
import os
from cudaq_einsum import EinsumSidecar

def main():
    print("=" * 60)
    print("Testing Sidecar Ctypes Approach")
    print("=" * 60)

    # Step 1: Load library via EinsumSidecar
    print("\n[1] Loading einsum library...")
    try:
        sidecar = EinsumSidecar()
        print(f"    Library loaded: {sidecar.lib_path}")
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
    sidecar.clear()

    result = cudaq.sample(ghz_kernel, shots_count=10)
    print(f"    Sample result: {result}")

    # Step 5: Retrieve sidecar data
    print("\n[5] Retrieving data from sidecar buffer...")
    json_str = sidecar.get_circuit_json()

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
