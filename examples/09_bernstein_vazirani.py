#!/usr/bin/env python3
"""
Example 09: Bernstein-Vazirani Algorithm

The Bernstein-Vazirani algorithm determines a hidden bitstring s in a
single query to a quantum oracle.  Given f(x) = s . x (mod 2), a single
evaluation of the circuit reveals the full string s.

This is a good test for the einsum backend because:
- It scales to many qubits easily (only H + CX + H)
- The answer is deterministic and easy to verify
- The circuit is shallow, so contraction is fast even at 20+ qubits

Key features:
- Hidden bitstring recovery for 4, 8, 12, 16 qubits
- Correctness verification
- Contraction path analysis at scale
"""

import time
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
import opt_einsum as oe


# ---------------------------------------------------------------------------
# BV Kernels
#
# Convention: hidden string s is read MSB-first, matching qubit ordering.
# s[j] = 1 means CX(q[j], ancilla).
#
# CUDA-Q state vector: index i in binary (MSB first) = q[0] q[1] ... q[N-1].
# Data qubits: q[0..n-1].  Ancilla: q[n] (LSB of state index).
# ---------------------------------------------------------------------------

@cudaq.kernel
def bv_4bit():
    """BV oracle for s = 1011  (4 data qubits + 1 ancilla)."""
    q = cudaq.qvector(5)
    x(q[4]); h(q[4])                            # ancilla in |->
    h(q[0]); h(q[1]); h(q[2]); h(q[3])          # data in |+>

    # Oracle: CX from q[j] to ancilla where s[j]=1
    # s = 1 0 1 1
    #     ^q0 ^q1 ^q2 ^q3
    cx(q[0], q[4])   # s[0] = 1
    # skip q[1]       # s[1] = 0
    cx(q[2], q[4])   # s[2] = 1
    cx(q[3], q[4])   # s[3] = 1

    h(q[0]); h(q[1]); h(q[2]); h(q[3])          # data back


@cudaq.kernel
def bv_8bit():
    """BV oracle for s = 10110101  (8 data + 1 ancilla)."""
    q = cudaq.qvector(9)
    x(q[8]); h(q[8])
    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])

    # s = 1 0 1 1 0 1 0 1
    cx(q[0], q[8])   # 1
    # q[1]             # 0
    cx(q[2], q[8])   # 1
    cx(q[3], q[8])   # 1
    # q[4]             # 0
    cx(q[5], q[8])   # 1
    # q[6]             # 0
    cx(q[7], q[8])   # 1

    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])


@cudaq.kernel
def bv_12bit():
    """BV oracle for s = 110100101011  (12 data + 1 ancilla)."""
    q = cudaq.qvector(13)
    x(q[12]); h(q[12])
    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])
    h(q[8]); h(q[9]); h(q[10]); h(q[11])

    # s = 1 1 0 1 0 0 1 0 1 0 1 1
    cx(q[0], q[12])   # 1
    cx(q[1], q[12])   # 1
    # q[2]             # 0
    cx(q[3], q[12])   # 1
    # q[4]             # 0
    # q[5]             # 0
    cx(q[6], q[12])   # 1
    # q[7]             # 0
    cx(q[8], q[12])   # 1
    # q[9]             # 0
    cx(q[10], q[12])  # 1
    cx(q[11], q[12])  # 1

    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])
    h(q[8]); h(q[9]); h(q[10]); h(q[11])


@cudaq.kernel
def bv_16bit():
    """BV oracle for s = 1011010110110101  (16 data + 1 ancilla)."""
    q = cudaq.qvector(17)
    x(q[16]); h(q[16])
    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])
    h(q[8]); h(q[9]); h(q[10]); h(q[11])
    h(q[12]); h(q[13]); h(q[14]); h(q[15])

    # s = 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1
    cx(q[0], q[16])   # 1
    # q[1]             # 0
    cx(q[2], q[16])   # 1
    cx(q[3], q[16])   # 1
    # q[4]             # 0
    cx(q[5], q[16])   # 1
    # q[6]             # 0
    cx(q[7], q[16])   # 1
    cx(q[8], q[16])   # 1
    # q[9]             # 0
    cx(q[10], q[16])  # 1
    cx(q[11], q[16])  # 1
    # q[12]            # 0
    cx(q[13], q[16])  # 1
    # q[14]            # 0
    cx(q[15], q[16])  # 1

    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])
    h(q[8]); h(q[9]); h(q[10]); h(q[11])
    h(q[12]); h(q[13]); h(q[14]); h(q[15])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_hidden_string(state: np.ndarray, n_data: int) -> str:
    """
    Extract the hidden string from the BV output state.

    Data qubits are q[0..n_data-1] (most significant bits in state index).
    Ancilla is q[n_data] (least significant bit).
    We marginalize over the ancilla and find the peak data state.
    """
    probs = np.abs(state) ** 2
    n_data_states = 2 ** n_data

    # Marginalize over ancilla (LSB): data_idx = state_idx >> 1
    data_probs = np.zeros(n_data_states)
    for idx in range(len(state)):
        data_idx = idx >> 1   # shift out the ancilla (LSB)
        data_probs[data_idx] += probs[idx]

    best_idx = np.argmax(data_probs)
    return format(best_idx, f'0{n_data}b')


def main():
    print("=" * 70)
    print("Example 09: Bernstein-Vazirani Algorithm")
    print("=" * 70)

    tests = [
        ("bv_4bit",  bv_4bit,  4,  5,  "1011"),
        ("bv_8bit",  bv_8bit,  8,  9,  "10110101"),
        ("bv_12bit", bv_12bit, 12, 13, "110100101011"),
        ("bv_16bit", bv_16bit, 16, 17, "1011010110110101"),
    ]

    # -------------------------------------------------------------------
    # [1] Run each BV circuit and verify
    # -------------------------------------------------------------------
    print(f"\n[1] Bernstein-Vazirani: recover hidden bitstring")
    print(f"    {'name':>10s}  {'qubits':>6s}  {'gates':>5s}  {'capture':>8s}  "
          f"{'contract':>9s}  {'hidden_s':>18s}  {'found':>18s}  {'ok':>4s}")
    print("    " + "-" * 90)

    for name, kernel, n_data, n_total, expected_s in tests:
        t0 = time.time()
        circuit = capture_circuit(kernel)
        t_cap = time.time() - t0

        t0 = time.time()
        args = circuit.to_einsum_sublist_args()
        state = oe.contract(*args).flatten()
        t_con = time.time() - t0

        found_s = extract_hidden_string(state, n_data)
        ok = (found_s == expected_s)

        print(f"    {name:>10s}  {n_total:6d}  {len(circuit.gates):5d}  "
              f"{t_cap*1000:6.1f}ms  {t_con*1000:7.1f}ms  {expected_s:>18s}  "
              f"{found_s:>18s}  {'pass' if ok else 'FAIL'}")

    # -------------------------------------------------------------------
    # [2] Contraction path analysis
    # -------------------------------------------------------------------
    print(f"\n[2] Contraction path analysis")
    for name, kernel, n_data, n_total, _ in tests:
        circuit = capture_circuit(kernel)
        args = circuit.to_einsum_sublist_args()
        path, info = oe.contract_path(*args, optimize='auto')
        print(f"    {name:>10s}: FLOPs={info.opt_cost:.2e}, "
              f"max_intermediate={int(info.largest_intermediate)}")

    # -------------------------------------------------------------------
    # [3] Detailed look at the 4-qubit case
    # -------------------------------------------------------------------
    print(f"\n[3] Detailed 4-qubit BV analysis")
    circuit = capture_circuit(bv_4bit)
    print(f"    Circuit: {circuit.num_qubits}q, {len(circuit.gates)}g, max_idx={circuit.max_index}")
    print(f"    Gate sequence:")
    for i, g in enumerate(circuit.gates):
        ctrl_str = f"ctrl={g.controls}" if g.controls else ""
        print(f"      {i:2d}: {g.name:4s} tgt={g.targets} {ctrl_str}")

    # Show state probabilities
    args = circuit.to_einsum_sublist_args()
    state = oe.contract(*args).flatten()
    probs = np.abs(state) ** 2
    print(f"\n    Non-zero amplitudes (prob > 0.01):")
    for idx in range(len(state)):
        if probs[idx] > 0.01:
            bs = format(idx, f'0{circuit.num_qubits}b')
            print(f"      |{bs}> : prob={probs[idx]:.4f}, amp={state[idx]:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
