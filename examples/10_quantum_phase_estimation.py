#!/usr/bin/env python3
"""
Example 10: Quantum Phase Estimation (QPE)

Quantum Phase Estimation is a core subroutine used in Shor's algorithm,
quantum chemistry, and HHL.  Given a unitary U and its eigenstate |u>,
QPE estimates the phase phi where U|u> = e^{2*pi*i*phi}|u>.

For diagonal unitaries (like T, S, Z gates) on the prepared eigenstate |1>,
the controlled-U reduces to a phase rotation on the control qubit:
    controlled-U^k |c>|1> = r1(k*theta, control) |c>|1>

This simplifies the circuit while testing the same einsum contraction
features (many gates, multi-qubit, controlled phases, inverse QFT).

Key features:
- Controlled phase kickback pattern
- Inverse QFT via controlled-phase decomposition
- Phase readout from measurement probabilities
- Verification against exact eigenvalues
"""

import math
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
import opt_einsum as oe


# ---------------------------------------------------------------------------
# 3-qubit inverse QFT block (used by all QPE kernels below)
#
# Standard iQFT on q[0], q[1], q[2] (MSB to LSB):
#   1. SWAP(q[0], q[2])        — bit reversal
#   2. H(q[2])
#   3. CP(-pi/2, ctrl=q[2], tgt=q[1])
#   4. H(q[1])
#   5. CP(-pi/4, ctrl=q[2], tgt=q[0])
#   6. CP(-pi/2, ctrl=q[1], tgt=q[0])
#   7. H(q[0])
#
# CP(theta, ctrl, tgt) decomposition:
#   r1(theta/2, ctrl); r1(theta/2, tgt); cx(ctrl, tgt);
#   r1(-theta/2, tgt); cx(ctrl, tgt)
# ---------------------------------------------------------------------------


@cudaq.kernel
def qpe_t_gate():
    """
    3-bit QPE for T gate:  T|1> = e^{i*pi/4}|1>,  phi = 1/8.
    Expected readout: |001> on counting register (value 1 = 1/8 * 8).
    """
    q = cudaq.qvector(4)
    x(q[3])                                     # eigenstate |1>
    h(q[0]); h(q[1]); h(q[2])                   # counting qubits

    # Phase kickback: r1(2^k * theta, q[n-1-k]) where theta = pi/4
    r1(4.0 * math.pi / 4, q[0])    # U^4: pi
    r1(2.0 * math.pi / 4, q[1])    # U^2: pi/2
    r1(1.0 * math.pi / 4, q[2])    # U^1: pi/4

    # Inverse QFT on q[0..2]
    swap(q[0], q[2])
    h(q[2])
    # CP(-pi/2, ctrl=q[2], tgt=q[1])
    r1(-math.pi / 4, q[2]); r1(-math.pi / 4, q[1])
    cx(q[2], q[1]); r1(math.pi / 4, q[1]); cx(q[2], q[1])
    h(q[1])
    # CP(-pi/4, ctrl=q[2], tgt=q[0])
    r1(-math.pi / 8, q[2]); r1(-math.pi / 8, q[0])
    cx(q[2], q[0]); r1(math.pi / 8, q[0]); cx(q[2], q[0])
    # CP(-pi/2, ctrl=q[1], tgt=q[0])
    r1(-math.pi / 4, q[1]); r1(-math.pi / 4, q[0])
    cx(q[1], q[0]); r1(math.pi / 4, q[0]); cx(q[1], q[0])
    h(q[0])


@cudaq.kernel
def qpe_s_gate():
    """
    3-bit QPE for S gate:  S|1> = e^{i*pi/2}|1>,  phi = 1/4.
    Expected readout: |010> (value 2 = 1/4 * 8).
    """
    q = cudaq.qvector(4)
    x(q[3])
    h(q[0]); h(q[1]); h(q[2])

    r1(4.0 * math.pi / 2, q[0])    # U^4: 2*pi ~ 0
    r1(2.0 * math.pi / 2, q[1])    # U^2: pi
    r1(1.0 * math.pi / 2, q[2])    # U^1: pi/2

    # iQFT
    swap(q[0], q[2])
    h(q[2])
    r1(-math.pi / 4, q[2]); r1(-math.pi / 4, q[1])
    cx(q[2], q[1]); r1(math.pi / 4, q[1]); cx(q[2], q[1])
    h(q[1])
    r1(-math.pi / 8, q[2]); r1(-math.pi / 8, q[0])
    cx(q[2], q[0]); r1(math.pi / 8, q[0]); cx(q[2], q[0])
    r1(-math.pi / 4, q[1]); r1(-math.pi / 4, q[0])
    cx(q[1], q[0]); r1(math.pi / 4, q[0]); cx(q[1], q[0])
    h(q[0])


@cudaq.kernel
def qpe_z_gate():
    """
    3-bit QPE for Z gate:  Z|1> = e^{i*pi}|1>,  phi = 1/2.
    Expected readout: |100> (value 4 = 1/2 * 8).
    """
    q = cudaq.qvector(4)
    x(q[3])
    h(q[0]); h(q[1]); h(q[2])

    r1(4.0 * math.pi, q[0])    # U^4: 4*pi ~ 0
    r1(2.0 * math.pi, q[1])    # U^2: 2*pi ~ 0
    r1(1.0 * math.pi, q[2])    # U^1: pi

    # iQFT
    swap(q[0], q[2])
    h(q[2])
    r1(-math.pi / 4, q[2]); r1(-math.pi / 4, q[1])
    cx(q[2], q[1]); r1(math.pi / 4, q[1]); cx(q[2], q[1])
    h(q[1])
    r1(-math.pi / 8, q[2]); r1(-math.pi / 8, q[0])
    cx(q[2], q[0]); r1(math.pi / 8, q[0]); cx(q[2], q[0])
    r1(-math.pi / 4, q[1]); r1(-math.pi / 4, q[0])
    cx(q[1], q[0]); r1(math.pi / 4, q[0]); cx(q[1], q[0])
    h(q[0])


@cudaq.kernel
def qpe_custom_phase():
    """
    3-bit QPE for custom gate with phi = 3/8.
    U|1> = e^{i * 2*pi * 3/8}|1> = e^{i * 3*pi/4}|1>.
    Expected readout: |011> (value 3 = 3/8 * 8).
    """
    q = cudaq.qvector(4)
    x(q[3])
    h(q[0]); h(q[1]); h(q[2])

    # theta = 2*pi * 3/8 = 3*pi/4
    r1(4.0 * 3.0 * math.pi / 4, q[0])   # U^4: 3*pi
    r1(2.0 * 3.0 * math.pi / 4, q[1])   # U^2: 3*pi/2
    r1(1.0 * 3.0 * math.pi / 4, q[2])   # U^1: 3*pi/4

    # iQFT
    swap(q[0], q[2])
    h(q[2])
    r1(-math.pi / 4, q[2]); r1(-math.pi / 4, q[1])
    cx(q[2], q[1]); r1(math.pi / 4, q[1]); cx(q[2], q[1])
    h(q[1])
    r1(-math.pi / 8, q[2]); r1(-math.pi / 8, q[0])
    cx(q[2], q[0]); r1(math.pi / 8, q[0]); cx(q[2], q[0])
    r1(-math.pi / 4, q[1]); r1(-math.pi / 4, q[0])
    cx(q[1], q[0]); r1(math.pi / 4, q[0]); cx(q[1], q[0])
    h(q[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_phase(state: np.ndarray, n_counting: int) -> tuple:
    """
    Extract the estimated phase from counting qubits.

    Counting qubits are q[0..n_counting-1] (MSBs of state index).
    Eigenstate is q[n_counting] (LSB).

    Returns (phase, counting_probs).
    """
    probs = np.abs(state) ** 2
    n_counting_states = 2 ** n_counting

    # Marginalize over eigenstate qubit (LSB)
    counting_probs = np.zeros(n_counting_states)
    for idx in range(len(state)):
        counting_idx = idx >> 1  # shift out eigenstate bit (LSB)
        counting_probs[counting_idx] += probs[idx]

    best_idx = np.argmax(counting_probs)
    phase = best_idx / n_counting_states
    return phase, counting_probs


def main():
    print("=" * 70)
    print("Example 10: Quantum Phase Estimation")
    print("=" * 70)

    # -------------------------------------------------------------------
    # [1] QPE for standard gates
    # -------------------------------------------------------------------
    tests = [
        ("T gate",  qpe_t_gate,       1/8, "001"),
        ("S gate",  qpe_s_gate,       1/4, "010"),
        ("Z gate",  qpe_z_gate,       1/2, "100"),
        ("phi=3/8", qpe_custom_phase, 3/8, "011"),
    ]
    n_counting = 3

    print(f"\n[1] 3-bit QPE results")
    print(f"    {'gate':>8s}  {'phi_exact':>10s}  {'phi_meas':>10s}  "
          f"{'register':>8s}  {'gates':>5s}  {'ok':>6s}")
    print("    " + "-" * 60)

    for name, kernel, expected_phase, expected_bits in tests:
        circuit = capture_circuit(kernel)
        args = circuit.to_einsum_sublist_args()
        state = oe.contract(*args).flatten()

        phase, counting_probs = extract_phase(state, n_counting)
        best_idx = np.argmax(counting_probs)
        measured_bits = format(best_idx, f'0{n_counting}b')
        ok = abs(phase - expected_phase) < 0.01 and counting_probs[best_idx] > 0.9

        print(f"    {name:>8s}  {expected_phase:10.4f}  {phase:10.4f}  "
              f"  |{measured_bits}>  {len(circuit.gates):5d}  {'pass' if ok else 'FAIL':>6s}")

    # -------------------------------------------------------------------
    # [2] Counting register probability distributions
    # -------------------------------------------------------------------
    print(f"\n[2] Counting register probabilities")

    for name, kernel, expected_phase, _ in tests:
        circuit = capture_circuit(kernel)
        args = circuit.to_einsum_sublist_args()
        state = oe.contract(*args).flatten()
        _, counting_probs = extract_phase(state, n_counting)

        print(f"\n    {name} (expected phi = {expected_phase}):")
        for idx in range(2 ** n_counting):
            if counting_probs[idx] > 0.001:
                phase_val = idx / (2 ** n_counting)
                bar_len = int(counting_probs[idx] * 40)
                print(f"      |{format(idx, f'0{n_counting}b')}> (phi={phase_val:.3f}): "
                      f"{'#' * bar_len} {counting_probs[idx]:.4f}")

    # -------------------------------------------------------------------
    # [3] Circuit structure analysis
    # -------------------------------------------------------------------
    print(f"\n[3] Circuit structure analysis")
    for name, kernel, _, _ in tests:
        circuit = capture_circuit(kernel)
        args = circuit.to_einsum_sublist_args()
        path, info = oe.contract_path(*args, optimize='auto')
        print(f"    {name:>8s}: {circuit.num_qubits}q, {len(circuit.gates)}g, "
              f"max_idx={circuit.max_index}, FLOPs={info.opt_cost:.2e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
