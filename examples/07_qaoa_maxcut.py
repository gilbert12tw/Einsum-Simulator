#!/usr/bin/env python3
"""
Example 07: QAOA for MaxCut

Demonstrates using the einsum tensor-network backend for the Quantum
Approximate Optimization Algorithm (QAOA) applied to the MaxCut problem.

Key features:
- Parameterized QAOA circuit with problem (gamma) and mixer (beta) layers
- Expectation value computation via tensor contraction
- Brute-force comparison to verify optimal solution
- Scaling from 4 to 8 qubits

NOTE: For circuits with many gates, opt_einsum is required for efficient
contraction. Raw numpy.einsum picks a terrible contraction path and hangs.
"""

import math
import numpy as np
import cudaq
from cudaq_einsum import capture_circuit
import opt_einsum as oe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def contract_state(circuit):
    """Contract circuit using opt_einsum (much faster than numpy.einsum)."""
    args = circuit.to_einsum_sublist_args()
    return oe.contract(*args).flatten()


def maxcut_cost(bitstring: int, edges: list) -> int:
    """Count the number of edges cut by a bitstring partition."""
    cost = 0
    for (u, v) in edges:
        if ((bitstring >> u) & 1) != ((bitstring >> v) & 1):
            cost += 1
    return cost


def maxcut_expectation(state: np.ndarray, edges: list) -> float:
    """Compute <C> = sum_x  p(x) * cut_value(x)."""
    probs = np.abs(state) ** 2
    return sum(probs[x] * maxcut_cost(x, edges) for x in range(len(state)))


def brute_force_maxcut(edges: list, n: int):
    """Find the maximum cut by brute force."""
    best_cost = -1
    best_strings = []
    for x in range(2**n):
        c = maxcut_cost(x, edges)
        if c > best_cost:
            best_cost = c
            best_strings = [x]
        elif c == best_cost:
            best_strings.append(x)
    return best_cost, best_strings


# ---------------------------------------------------------------------------
# QAOA kernels
# ---------------------------------------------------------------------------

@cudaq.kernel
def qaoa_4q(gamma: float, beta: float):
    """1-layer QAOA for 4-node ring graph: 0-1-2-3-0"""
    q = cudaq.qvector(4)
    h(q[0]); h(q[1]); h(q[2]); h(q[3])

    # Problem unitary — ZZ interaction per edge
    cx(q[0], q[1]); rz(2.0 * gamma, q[1]); cx(q[0], q[1])
    cx(q[1], q[2]); rz(2.0 * gamma, q[2]); cx(q[1], q[2])
    cx(q[2], q[3]); rz(2.0 * gamma, q[3]); cx(q[2], q[3])
    cx(q[3], q[0]); rz(2.0 * gamma, q[0]); cx(q[3], q[0])

    # Mixer unitary
    rx(2.0 * beta, q[0]); rx(2.0 * beta, q[1])
    rx(2.0 * beta, q[2]); rx(2.0 * beta, q[3])


@cudaq.kernel
def qaoa_6q(gamma: float, beta: float):
    """
    1-layer QAOA for 6-node graph:
        0 --- 1 --- 2
        |           |
        5 --- 4 --- 3
    Edges: (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,4),(1,5)
    """
    q = cudaq.qvector(6)
    h(q[0]); h(q[1]); h(q[2]); h(q[3]); h(q[4]); h(q[5])

    cx(q[0], q[1]); rz(2.0 * gamma, q[1]); cx(q[0], q[1])
    cx(q[1], q[2]); rz(2.0 * gamma, q[2]); cx(q[1], q[2])
    cx(q[2], q[3]); rz(2.0 * gamma, q[3]); cx(q[2], q[3])
    cx(q[3], q[4]); rz(2.0 * gamma, q[4]); cx(q[3], q[4])
    cx(q[4], q[5]); rz(2.0 * gamma, q[5]); cx(q[4], q[5])
    cx(q[5], q[0]); rz(2.0 * gamma, q[0]); cx(q[5], q[0])
    cx(q[0], q[4]); rz(2.0 * gamma, q[4]); cx(q[0], q[4])
    cx(q[1], q[5]); rz(2.0 * gamma, q[5]); cx(q[1], q[5])

    rx(2.0 * beta, q[0]); rx(2.0 * beta, q[1]); rx(2.0 * beta, q[2])
    rx(2.0 * beta, q[3]); rx(2.0 * beta, q[4]); rx(2.0 * beta, q[5])


@cudaq.kernel
def qaoa_8q(gamma: float, beta: float):
    """
    1-layer QAOA for 8-node 3-regular graph.
    Edges: ring(0..7) + cross(0-3, 1-6, 2-5, 4-7)
    """
    q = cudaq.qvector(8)
    h(q[0]); h(q[1]); h(q[2]); h(q[3])
    h(q[4]); h(q[5]); h(q[6]); h(q[7])

    cx(q[0], q[1]); rz(2.0 * gamma, q[1]); cx(q[0], q[1])
    cx(q[1], q[2]); rz(2.0 * gamma, q[2]); cx(q[1], q[2])
    cx(q[2], q[3]); rz(2.0 * gamma, q[3]); cx(q[2], q[3])
    cx(q[3], q[4]); rz(2.0 * gamma, q[4]); cx(q[3], q[4])
    cx(q[4], q[5]); rz(2.0 * gamma, q[5]); cx(q[4], q[5])
    cx(q[5], q[6]); rz(2.0 * gamma, q[6]); cx(q[5], q[6])
    cx(q[6], q[7]); rz(2.0 * gamma, q[7]); cx(q[6], q[7])
    cx(q[7], q[0]); rz(2.0 * gamma, q[0]); cx(q[7], q[0])
    cx(q[0], q[3]); rz(2.0 * gamma, q[3]); cx(q[0], q[3])
    cx(q[1], q[6]); rz(2.0 * gamma, q[6]); cx(q[1], q[6])
    cx(q[2], q[5]); rz(2.0 * gamma, q[5]); cx(q[2], q[5])
    cx(q[4], q[7]); rz(2.0 * gamma, q[7]); cx(q[4], q[7])

    rx(2.0 * beta, q[0]); rx(2.0 * beta, q[1])
    rx(2.0 * beta, q[2]); rx(2.0 * beta, q[3])
    rx(2.0 * beta, q[4]); rx(2.0 * beta, q[5])
    rx(2.0 * beta, q[6]); rx(2.0 * beta, q[7])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Example 07: QAOA for MaxCut")
    print("=" * 70)

    # -------------------------------------------------------------------
    # [1]  4-qubit ring graph
    # -------------------------------------------------------------------
    edges_4 = [(0,1), (1,2), (2,3), (3,0)]
    n4 = 4
    best_cost_4, best_strs_4 = brute_force_maxcut(edges_4, n4)

    print(f"\n[1] 4-qubit ring graph  (edges: {edges_4})")
    print(f"    Brute-force MaxCut = {best_cost_4}")
    print(f"    Optimal partitions: {[format(s, f'0{n4}b') for s in best_strs_4]}")

    # Grid search over (gamma, beta)
    grid_n = 15
    print(f"\n    Grid search ({grid_n}x{grid_n} = {grid_n**2} points) ...")
    gammas = np.linspace(0, math.pi, grid_n)
    betas  = np.linspace(0, math.pi / 2, grid_n)
    best_exp = -1.0
    best_gb = (0.0, 0.0)

    for gamma in gammas:
        for beta in betas:
            circuit = capture_circuit(qaoa_4q, gamma, beta)
            state = contract_state(circuit)
            exp_val = maxcut_expectation(state, edges_4)
            if exp_val > best_exp:
                best_exp = exp_val
                best_gb = (gamma, beta)

    print(f"    Best <C>  = {best_exp:.4f}  (optimal = {best_cost_4})")
    print(f"    Best (gamma, beta) = ({best_gb[0]:.4f}, {best_gb[1]:.4f})")
    print(f"    Approximation ratio = {best_exp / best_cost_4:.4f}")

    # Probability distribution at best parameters
    circuit = capture_circuit(qaoa_4q, *best_gb)
    state = contract_state(circuit)
    probs = np.abs(state) ** 2
    print(f"\n    Top-5 most probable bitstrings:")
    top_idx = np.argsort(probs)[::-1][:5]
    for idx in top_idx:
        bs = format(idx, f'0{n4}b')
        cut = maxcut_cost(idx, edges_4)
        print(f"      |{bs}> : prob={probs[idx]:.4f}, cut={cut}")

    # -------------------------------------------------------------------
    # [2]  6-qubit graph
    # -------------------------------------------------------------------
    edges_6 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,4),(1,5)]
    n6 = 6
    best_cost_6, _ = brute_force_maxcut(edges_6, n6)

    print(f"\n[2] 6-qubit graph  ({len(edges_6)} edges)")
    print(f"    Brute-force MaxCut = {best_cost_6}")

    grid_n6 = 11
    best_exp6 = -1.0
    best_gb6 = (0.0, 0.0)
    for gamma in np.linspace(0, math.pi, grid_n6):
        for beta in np.linspace(0, math.pi / 2, grid_n6):
            circuit = capture_circuit(qaoa_6q, gamma, beta)
            state = contract_state(circuit)
            exp_val = maxcut_expectation(state, edges_6)
            if exp_val > best_exp6:
                best_exp6 = exp_val
                best_gb6 = (gamma, beta)

    print(f"    Best <C>  = {best_exp6:.4f}")
    print(f"    Approximation ratio = {best_exp6 / best_cost_6:.4f}")

    # -------------------------------------------------------------------
    # [3]  8-qubit graph — single capture + contraction analysis
    # -------------------------------------------------------------------
    edges_8 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
               (0,3),(1,6),(2,5),(4,7)]
    n8 = 8
    best_cost_8, _ = brute_force_maxcut(edges_8, n8)

    print(f"\n[3] 8-qubit 3-regular graph  ({len(edges_8)} edges)")
    print(f"    Brute-force MaxCut = {best_cost_8}")

    circuit8 = capture_circuit(qaoa_8q, 0.5, 0.3)
    print(f"    Circuit: {circuit8.num_qubits}q, {len(circuit8.gates)}g, max_idx={circuit8.max_index}")

    # Use sublist format — 8q QAOA exceeds the 52-char subscript limit
    args = circuit8.to_einsum_sublist_args()
    path, info = oe.contract_path(*args, optimize='auto')
    print(f"    Contraction FLOPs: {info.opt_cost:.2e}")
    print(f"    Largest intermediate: {info.largest_intermediate} elements")

    # Quick scan
    grid_n8 = 9
    best_exp8 = -1.0
    for gamma in np.linspace(0, math.pi, grid_n8):
        for beta in np.linspace(0, math.pi / 2, grid_n8):
            circuit = capture_circuit(qaoa_8q, gamma, beta)
            state = contract_state(circuit)
            exp_val = maxcut_expectation(state, edges_8)
            if exp_val > best_exp8:
                best_exp8 = exp_val

    print(f"    Best <C>  = {best_exp8:.4f}")
    print(f"    Approximation ratio = {best_exp8 / best_cost_8:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
