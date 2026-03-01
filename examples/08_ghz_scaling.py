#!/usr/bin/env python3
"""
Example 08: GHZ State Scaling Benchmark

Benchmarks the einsum tensor-network capture and contraction for GHZ
states from 4 to 20+ qubits.  Compares numpy.einsum vs opt_einsum
and reports contraction path statistics.

Key features:
- Scaling from small (4q) to large (20q) circuits
- Contraction path analysis (FLOPs, memory, path)
- Correctness verification against known GHZ amplitudes
- Timing breakdown: capture vs contraction
"""

import time
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
import opt_einsum as oe


@cudaq.kernel
def ghz(n: int):
    """N-qubit GHZ state:  (|00...0> + |11...1>) / sqrt(2)"""
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(1, n):
        cx(q[0], q[i])


def verify_ghz(state: np.ndarray, n: int) -> bool:
    """Check that the state is a valid GHZ state."""
    expected_amp = 1.0 / np.sqrt(2)
    ok = (abs(abs(state[0]) - expected_amp) < 1e-4 and
          abs(abs(state[-1]) - expected_amp) < 1e-4 and
          abs(np.linalg.norm(state) - 1.0) < 1e-4)
    # All middle amplitudes should be zero
    if len(state) > 2:
        ok = ok and np.allclose(state[1:-1], 0, atol=1e-6)
    return ok


def main():
    print("=" * 70)
    print("Example 08: GHZ State Scaling Benchmark")
    print("=" * 70)

    # -------------------------------------------------------------------
    # [1] Basic scaling
    # -------------------------------------------------------------------
    print(f"\n[1] Capture + contraction scaling (opt_einsum)")
    print(f"    {'n':>3s}  {'gates':>5s}  {'idx':>4s}  {'capture':>9s}  {'contract':>10s}  {'sv_len':>8s}  {'ok':>4s}")
    print("    " + "-" * 55)

    qubit_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    timings = []

    for n in qubit_counts:
        # Capture
        t0 = time.time()
        circuit = capture_circuit(ghz, n)
        t_cap = time.time() - t0

        # Contract with opt_einsum
        t0 = time.time()
        args = circuit.to_einsum_sublist_args()
        state = oe.contract(*args).flatten()
        t_con = time.time() - t0

        ok = verify_ghz(state, n)
        timings.append((n, t_cap, t_con))

        print(f"    {n:3d}  {len(circuit.gates):5d}  {circuit.max_index:4d}  "
              f"{t_cap*1000:7.1f}ms  {t_con*1000:8.1f}ms  {len(state):8d}  "
              f"{'pass' if ok else 'FAIL'}")

    # -------------------------------------------------------------------
    # [2] Contraction path analysis
    # -------------------------------------------------------------------
    print(f"\n[2] Contraction path analysis")
    print(f"    {'n':>3s}  {'operands':>8s}  {'FLOPs':>12s}  {'max_interm':>12s}  {'strategy':>10s}")
    print("    " + "-" * 55)

    for n in [4, 8, 12, 16, 20]:
        circuit = capture_circuit(ghz, n)
        args = circuit.to_einsum_sublist_args()
        n_operands = (len(args) - 1) // 2
        path, info = oe.contract_path(*args, optimize='auto')
        print(f"    {n:3d}  {n_operands:8d}  {info.opt_cost:12.2e}  "
              f"{int(info.largest_intermediate):12d}  {'auto':>10s}")

    # -------------------------------------------------------------------
    # [3] Compare contraction strategies
    # -------------------------------------------------------------------
    print(f"\n[3] Contraction strategy comparison (12-qubit GHZ)")

    circuit = capture_circuit(ghz, 12)
    args = circuit.to_einsum_sublist_args()

    strategies = ['auto', 'greedy']
    for strat in strategies:
        t0 = time.time()
        path, info = oe.contract_path(*args, optimize=strat)
        t_path = time.time() - t0

        t0 = time.time()
        result = oe.contract(*args, optimize=strat)
        t_total = time.time() - t0

        print(f"    {strat:>10s}: FLOPs={info.opt_cost:.2e}, "
              f"path_time={t_path*1000:.1f}ms, total={t_total*1000:.1f}ms")

    # -------------------------------------------------------------------
    # [4] Push the limits — larger qubits (capture only, no full SV)
    # -------------------------------------------------------------------
    print(f"\n[4] Large-scale capture (structure only, no full state vector)")
    print(f"    {'n':>3s}  {'gates':>5s}  {'indices':>7s}  {'capture':>9s}  {'FLOPs':>12s}")
    print("    " + "-" * 48)

    for n in [20, 22, 24]:
        t0 = time.time()
        circuit = capture_circuit(ghz, n)
        t_cap = time.time() - t0

        # Path analysis only (no actual contraction — state vector too large)
        args = circuit.to_einsum_sublist_args()
        path, info = oe.contract_path(*args, optimize='auto')

        print(f"    {n:3d}  {len(circuit.gates):5d}  {circuit.max_index:7d}  "
              f"{t_cap*1000:7.1f}ms  {info.opt_cost:12.2e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
