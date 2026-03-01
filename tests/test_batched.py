#!/usr/bin/env python3
"""
Thorough functional tests for the batched evaluation API.

Test sections:
  1. make_rotation_matrices — math correctness, shapes, edge cases
  2. get_parametric_gate_info — identification, validation, error paths
  3. build_batched_args / batched_contract — shapes, correctness, error paths
  4. Numerical correctness — batch vs sequential, normalization, amplitude vs sv
  5. End-to-end — comparison against benchmark's reference implementation, QSVM

Run with: pytest tests/test_batched.py -v
"""

import json
import math
import numpy as np
import pytest

from cudaq_einsum.einsum_circuit import (
    EinsumCircuit, PARAMETRIC_GATE_CODES, ParametricGateInfo,
)
from cudaq_einsum.batched import (
    make_rotation_matrices, build_batched_args, batched_contract,
    _basis_bras,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cudaq
    from cudaq_einsum import capture_circuit
    HAS_CUDAQ = True
except Exception:
    HAS_CUDAQ = False


# ============================================================================
# Helpers
# ============================================================================

def _make_circuit_json(gates, num_qubits=2):
    """
    Build minimal EinsumCircuit JSON from a list of (name, params, controls, targets).
    Supports up to num_qubits qubits, tracked independently.
    """
    current_idx = list(range(num_qubits))
    max_idx = num_qubits - 1
    gate_list = []

    for name, params, controls, targets in gates:
        out_idx_list = []
        in_idx_list = [current_idx[t] for t in targets]
        for t in targets:
            max_idx += 1
            current_idx[t] = max_idx
            out_idx_list.append(max_idx)

        dim = 2 ** len(targets)
        # Identity matrix as placeholder
        flat_matrix = [
            [1 if r == c else 0, 0]
            for r in range(dim) for c in range(dim)
        ]
        gate_list.append({
            'name': name,
            'params': params,
            'controls': controls,
            'targets': targets,
            'inputIndices': in_idx_list,
            'outputIndices': out_idx_list,
            'matrix': flat_matrix,
        })

    return json.dumps({
        'numQubits': num_qubits,
        'numGates': len(gate_list),
        'maxIndex': max_idx,
        'initialStates': [{'qubitId': i, 'index': i} for i in range(num_qubits)],
        'gates': gate_list,
        'outputIndices': current_idx,
    })


# ============================================================================
# 1. make_rotation_matrices — math correctness
# ============================================================================

class TestMakeRotationMatrices:

    # ── Gate formulas ─────────────────────────────────────────────────────────

    def test_rx_formula(self):
        """RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]"""
        for theta in [0.0, math.pi / 6, math.pi / 2, math.pi, 2 * math.pi]:
            result = make_rotation_matrices(np.array([[theta]]), np.array([0]))
            mat = result[0, 0]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            expected = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            np.testing.assert_allclose(mat, expected, atol=1e-14,
                                       err_msg=f"RX({theta:.4f}) failed")

    def test_ry_formula(self):
        """RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
        for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
            result = make_rotation_matrices(np.array([[theta]]), np.array([1]))
            mat = result[0, 0]
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            expected = np.array([[c, -s], [s, c]], dtype=np.complex128)
            np.testing.assert_allclose(mat, expected, atol=1e-14,
                                       err_msg=f"RY({theta:.4f}) failed")

    def test_rz_formula(self):
        """RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2})"""
        for theta in [0.0, math.pi / 3, math.pi, 3 * math.pi / 2]:
            result = make_rotation_matrices(np.array([[theta]]), np.array([2]))
            mat = result[0, 0]
            expected = np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                dtype=np.complex128
            )
            np.testing.assert_allclose(mat, expected, atol=1e-14,
                                       err_msg=f"RZ({theta:.4f}) failed")

    def test_p_formula(self):
        """P(θ) = [[1, 0], [0, e^{iθ}]]"""
        for theta in [0.0, math.pi / 5, math.pi, 2 * math.pi]:
            result = make_rotation_matrices(np.array([[theta]]), np.array([3]))
            mat = result[0, 0]
            expected = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
            np.testing.assert_allclose(mat, expected, atol=1e-14,
                                       err_msg=f"P({theta:.4f}) failed")

    # ── Special angles ────────────────────────────────────────────────────────

    def test_all_identity_at_zero(self):
        """All gate types at θ=0 must produce the identity matrix."""
        codes = np.array([0, 1, 2, 3])
        result = make_rotation_matrices(np.zeros((1, 4)), codes)
        eye = np.eye(2, dtype=np.complex128)
        for j, name in enumerate(['rx', 'ry', 'rz', 'p']):
            np.testing.assert_allclose(result[0, j], eye, atol=1e-14,
                                       err_msg=f"{name}(0) != I")

    def test_rx_pi_is_X_up_to_phase(self):
        """RX(π) = -i·X — should be proportional to Pauli X."""
        result = make_rotation_matrices(np.array([[math.pi]]), np.array([0]))
        mat = result[0, 0]
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        # Check that mat = -i * X (global phase)
        np.testing.assert_allclose(mat, -1j * X, atol=1e-14)

    def test_ry_pi_state_flip(self):
        """RY(π)|0⟩ = |1⟩."""
        result = make_rotation_matrices(np.array([[math.pi]]), np.array([1]))
        mat = result[0, 0]
        zero = np.array([1, 0], dtype=np.complex128)
        one = np.array([0, 1], dtype=np.complex128)
        np.testing.assert_allclose(mat @ zero, one, atol=1e-14)

    def test_rz_pi_diagonal(self):
        """RZ(π) = diag(-i, i)."""
        result = make_rotation_matrices(np.array([[math.pi]]), np.array([2]))
        mat = result[0, 0]
        expected = np.diag(np.array([-1j, 1j], dtype=np.complex128))
        np.testing.assert_allclose(mat, expected, atol=1e-14)

    # ── Unitarity ─────────────────────────────────────────────────────────────

    def test_all_matrices_are_unitary(self):
        """U† U = I for all gate types and random angles."""
        rng = np.random.default_rng(42)
        B, P = 20, 8
        thetas = rng.uniform(-2 * math.pi, 2 * math.pi, (B, P))
        codes = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        mats = make_rotation_matrices(thetas, codes)  # (B, P, 2, 2)
        eye = np.eye(2, dtype=np.complex128)
        for b in range(B):
            for p in range(P):
                m = mats[b, p]
                np.testing.assert_allclose(m.conj().T @ m, eye, atol=1e-13,
                                           err_msg=f"Non-unitary at b={b}, p={p}")

    # ── Shape contracts ───────────────────────────────────────────────────────

    def test_batch_shape(self):
        """Output shape is (B, P, 2, 2)."""
        B, P = 7, 5
        thetas = np.random.randn(B, P)
        codes = np.array([0, 1, 2, 3, 0])
        result = make_rotation_matrices(thetas, codes)
        assert result.shape == (B, P, 2, 2)

    def test_different_codes_different_matrices(self):
        """Different gate codes must produce distinct matrices for the same angle."""
        theta = 1.23
        codes = np.array([0, 1, 2, 3])
        result = make_rotation_matrices(np.array([[theta] * 4]), codes)
        mats = [result[0, j] for j in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.allclose(mats[i], mats[j]), \
                    f"code {i} and code {j} produced same matrix"

    # ── Torch backend ─────────────────────────────────────────────────────────

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_torch_matches_numpy_exactly(self):
        """Torch and numpy backends must be numerically identical."""
        rng = np.random.default_rng(0)
        B, P = 6, 4
        thetas = rng.standard_normal((B, P))
        codes = np.array([0, 1, 2, 3])
        np_result = make_rotation_matrices(thetas, codes, backend='numpy')
        torch_result = make_rotation_matrices(thetas, codes, backend='torch')
        np.testing.assert_allclose(
            torch_result.cpu().numpy(), np_result, atol=1e-13
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_torch_tensor_input(self):
        """Torch tensors can be passed directly as thetas/codes."""
        thetas_t = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        codes_t = torch.tensor([0, 2], dtype=torch.long)
        result = make_rotation_matrices(thetas_t, codes_t, backend='torch')
        assert result.dtype == torch.complex128
        assert result.shape == (1, 2, 2, 2)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_torch_unitary(self):
        """Torch backend matrices are also unitary."""
        B, P = 5, 3
        thetas = np.random.randn(B, P)
        codes = np.array([0, 1, 2])
        mats = make_rotation_matrices(thetas, codes, backend='torch')
        eye = torch.eye(2, dtype=torch.complex128)
        for b in range(B):
            for p in range(P):
                m = mats[b, p]
                product = m.conj().T @ m
                assert torch.allclose(product, eye, atol=1e-13)


# ============================================================================
# 2. get_parametric_gate_info
# ============================================================================

class TestGetParametricGateInfo:

    def test_identifies_rx_ry_rz_p(self):
        """Should find all parametric gate types by default."""
        json_str = _make_circuit_json([
            ('h', [], [], [0]),
            ('rx', [0.5], [], [0]),
            ('ry', [0.3], [], [1]),
            ('rz', [0.7], [], [0]),
            ('p', [1.2], [], [1]),
        ])
        circuit = EinsumCircuit.from_json(json_str)
        positions, codes, info = circuit.get_parametric_gate_info()
        assert positions == [1, 2, 3, 4]
        assert codes == [0, 1, 2, 3]
        assert [inf.gate_name for inf in info] == ['rx', 'ry', 'rz', 'p']

    def test_preserves_original_params(self):
        """original_param field should match the gate's first param."""
        params_in = [0.123, -0.456, 1.789]
        json_str = _make_circuit_json([
            ('rx', [params_in[0]], [], [0]),
            ('ry', [params_in[1]], [], [1]),
            ('rz', [params_in[2]], [], [0]),
        ])
        circuit = EinsumCircuit.from_json(json_str)
        _, _, info = circuit.get_parametric_gate_info()
        for inf, expected_p in zip(info, params_in):
            assert inf.original_param == pytest.approx(expected_p)

    def test_position_matches_gate_index(self):
        """position field must equal the gate's index in circuit.gates."""
        json_str = _make_circuit_json([
            ('h', [], [], [0]),
            ('h', [], [], [1]),
            ('rx', [0.5], [], [0]),
            ('cx', [], [0], [1]),
            ('rz', [0.7], [], [1]),
        ])
        circuit = EinsumCircuit.from_json(json_str)
        positions, _, info = circuit.get_parametric_gate_info(gate_names=['rx', 'rz'])
        assert positions == [2, 4]
        assert info[0].position == 2
        assert info[1].position == 4

    def test_qubit_field_correct(self):
        """qubit field should match gate target qubit."""
        json_str = _make_circuit_json([
            ('rx', [0.5], [], [1]),   # targets qubit 1
            ('ry', [0.3], [], [0]),   # targets qubit 0
        ])
        circuit = EinsumCircuit.from_json(json_str)
        _, _, info = circuit.get_parametric_gate_info()
        assert info[0].qubit == 1
        assert info[1].qubit == 0

    def test_gate_name_filter(self):
        """Custom gate_names filter should work."""
        json_str = _make_circuit_json([
            ('rx', [0.5], [], [0]),
            ('ry', [0.3], [], [1]),
            ('rz', [0.7], [], [0]),
        ])
        circuit = EinsumCircuit.from_json(json_str)
        positions, codes, _ = circuit.get_parametric_gate_info(gate_names=['rz'])
        assert positions == [2]
        assert codes == [2]

    def test_no_parametric_gates_returns_empty(self):
        """Should return empty lists when no parametric gates exist."""
        json_str = _make_circuit_json([('h', [], [], [0]), ('h', [], [], [1])])
        circuit = EinsumCircuit.from_json(json_str)
        positions, codes, info = circuit.get_parametric_gate_info()
        assert positions == []
        assert codes == []
        assert info == []

    def test_code_constants(self):
        """PARAMETRIC_GATE_CODES must have correct values."""
        assert PARAMETRIC_GATE_CODES['rx'] == 0
        assert PARAMETRIC_GATE_CODES['ry'] == 1
        assert PARAMETRIC_GATE_CODES['rz'] == 2
        assert PARAMETRIC_GATE_CODES['r1'] == 3
        assert PARAMETRIC_GATE_CODES['p'] == 3

    def test_raises_on_controlled_parametric(self):
        """Should raise ValueError when a parametric gate has controls."""
        json_str = _make_circuit_json([
            ('rx', [0.5], [0], [1]),  # controlled rx — invalid
        ])
        circuit = EinsumCircuit.from_json(json_str)
        with pytest.raises(ValueError, match="controlled gate"):
            circuit.get_parametric_gate_info()

    def test_raises_on_multi_target_parametric(self):
        """Should raise ValueError when a parametric gate has >1 target."""
        json_str = _make_circuit_json([
            ('rx', [0.5], [], [0, 1]),  # two-target rx — invalid
        ])
        circuit = EinsumCircuit.from_json(json_str)
        with pytest.raises(ValueError, match="2 targets"):
            circuit.get_parametric_gate_info()

    def test_returns_parametric_gate_info_instances(self):
        """info_list items should be ParametricGateInfo instances."""
        json_str = _make_circuit_json([('ry', [0.5], [], [0])])
        circuit = EinsumCircuit.from_json(json_str)
        _, _, info = circuit.get_parametric_gate_info()
        assert len(info) == 1
        assert isinstance(info[0], ParametricGateInfo)


# ============================================================================
# 3. build_batched_args / batched_contract — shapes & error paths
# ============================================================================

@pytest.mark.skipif(not HAS_CUDAQ, reason="CUDA-Q not available")
class TestBuildBatchedArgsShapesAndErrors:

    @staticmethod
    def _simple_circuit():
        """2-qubit: H, RX, RY, CX."""
        @cudaq.kernel
        def simple(t0: float, t1: float):
            q = cudaq.qvector(2)
            h(q[0])
            rx(t0, q[0])
            ry(t1, q[1])
            cx(q[0], q[1])
        return capture_circuit(simple, 0.0, 0.0)

    def test_amplitude_mode_shape(self):
        """amplitude mode → (B,)."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        B = 5
        rot = make_rotation_matrices(
            np.random.randn(B, len(positions)), np.array(codes_list)
        )
        args = build_batched_args(circuit, positions, rot, mode='amplitude')
        out = np.einsum(*args)
        assert out.shape == (B,)
        assert out.dtype == np.complex128

    def test_statevector_mode_shape(self):
        """statevector mode → (B, 2^n)."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        B = 3
        rot = make_rotation_matrices(
            np.random.randn(B, len(positions)), np.array(codes_list)
        )
        args = build_batched_args(circuit, positions, rot, mode='statevector')
        out = np.einsum(*args)
        assert out.shape == (B,) + (2,) * circuit.num_qubits

    def test_raises_on_unknown_mode(self):
        """Unknown mode raises ValueError."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        rot = make_rotation_matrices(
            np.zeros((1, len(positions))), np.array(codes_list)
        )
        with pytest.raises(ValueError, match="Unknown mode"):
            build_batched_args(circuit, positions, rot, mode='bogus')

    def test_raises_on_empty_param_positions(self):
        """Empty param_positions raises ValueError."""
        circuit = self._simple_circuit()
        rot = np.empty((2, 0, 2, 2), dtype=np.complex128)
        with pytest.raises(ValueError, match="empty"):
            build_batched_args(circuit, [], rot, mode='amplitude')

    def test_raises_on_param_count_mismatch(self):
        """Mismatch between param_positions length and rot_mats dim raises."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        # pass rot_mats with wrong number of params
        rot = make_rotation_matrices(
            np.zeros((2, len(positions) + 1)),
            np.array(codes_list + [0])
        )
        with pytest.raises(ValueError, match="mismatch"):
            build_batched_args(circuit, positions, rot, mode='amplitude')

    def test_batched_contract_returns_correct_type(self):
        """batched_contract returns numpy array for numpy rot_mats."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        rot = make_rotation_matrices(
            np.random.randn(4, len(positions)), np.array(codes_list)
        )
        result = batched_contract(circuit, positions, rot, mode='amplitude')
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_torch_auto_detected(self):
        """build_batched_args with torch rot_mats produces all torch tensors."""
        circuit = self._simple_circuit()
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        rot = make_rotation_matrices(
            np.random.randn(3, len(positions)), np.array(codes_list),
            backend='torch'
        )
        args = build_batched_args(circuit, positions, rot, mode='amplitude')
        for item in args:
            if not isinstance(item, list):
                assert isinstance(item, torch.Tensor), \
                    f"Non-torch item in args: {type(item)}"
        result = torch.einsum(*args)
        assert result.shape == (3,)


# ============================================================================
# 4. Numerical correctness
# ============================================================================

@pytest.mark.skipif(not HAS_CUDAQ, reason="CUDA-Q not available")
class TestNumericalCorrectness:

    # ── Batch vs. sequential state vectors ───────────────────────────────────

    def test_statevector_batch_matches_sequential(self):
        """
        Each batch[i] state vector must equal the state vector from a separate
        single-parameter circuit capture.
        """
        @cudaq.kernel
        def ry_circuit(t: float):
            q = cudaq.qvector(2)
            ry(t, q[0])
            cx(q[0], q[1])

        theta_vals = [0.0, 0.3, 0.7, 1.2, math.pi / 2]
        B = len(theta_vals)

        # Sequential reference
        ref_svs = []
        for t in theta_vals:
            circuit_t = capture_circuit(ry_circuit, t)
            ref_svs.append(circuit_t.get_state_vector())

        # Batched
        circuit0 = capture_circuit(ry_circuit, 0.0)
        positions, codes_list, _ = circuit0.get_parametric_gate_info()
        codes = np.array(codes_list)
        thetas = np.array([[t] for t in theta_vals])
        rot = make_rotation_matrices(thetas, codes)
        args = build_batched_args(circuit0, positions, rot, mode='statevector')
        batch_svs = np.einsum(*args)

        for i, (ref, bat) in enumerate(zip(ref_svs, batch_svs.reshape(B, -1))):
            np.testing.assert_allclose(
                bat, ref, atol=1e-4,
                err_msg=f"State vector mismatch at batch index {i}"
            )

    def test_all_batched_svs_are_normalized(self):
        """Every state vector in a batch must have unit L2 norm."""
        @cudaq.kernel
        def rx_rz(t0: float, t1: float):
            q = cudaq.qvector(3)
            h(q[0])
            rx(t0, q[0])
            rz(t1, q[1])
            cx(q[0], q[2])

        circuit = capture_circuit(rx_rz, 0.0, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)
        B = 10
        thetas = np.random.default_rng(7).uniform(0, 2 * math.pi, (B, len(positions)))
        rot = make_rotation_matrices(thetas, codes)
        args = build_batched_args(circuit, positions, rot, mode='statevector')
        batch = np.einsum(*args).reshape(B, -1)
        norms = np.linalg.norm(batch, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # ── Amplitude mode vs. statevector element ────────────────────────────────

    def test_amplitude_zero_matches_sv_index_0(self):
        """Amplitude(target_state=0) == sv[0] for all batch items."""
        @cudaq.kernel
        def circ(t: float):
            q = cudaq.qvector(2)
            ry(t, q[0])
            cx(q[0], q[1])

        circuit = capture_circuit(circ, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)
        B = 6
        thetas = np.random.default_rng(1).uniform(0, math.pi, (B, len(positions)))
        rot = make_rotation_matrices(thetas, codes)

        amp_args = build_batched_args(circuit, positions, rot, mode='amplitude', target_state=0)
        sv_args = build_batched_args(circuit, positions, rot, mode='statevector')
        amps = np.einsum(*amp_args)
        svs = np.einsum(*sv_args).reshape(B, -1)

        np.testing.assert_allclose(amps, svs[:, 0], atol=1e-11)

    def test_amplitude_all_states_match_sv(self):
        """amplitude(target_state=k) == sv[k] for every k in 0..2^n-1."""
        @cudaq.kernel
        def circ(t: float):
            q = cudaq.qvector(2)
            ry(t, q[0])
            cx(q[0], q[1])

        circuit = capture_circuit(circ, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)
        theta_val = 0.8
        thetas = np.array([[theta_val]])
        rot = make_rotation_matrices(thetas, codes)

        sv_args = build_batched_args(circuit, positions, rot, mode='statevector')
        sv = np.einsum(*sv_args)[0].flatten()  # (4,)

        for k in range(4):
            amp_args = build_batched_args(
                circuit, positions, rot, mode='amplitude', target_state=k
            )
            amp = np.einsum(*amp_args)[0]
            np.testing.assert_allclose(
                amp, sv[k], atol=1e-11,
                err_msg=f"Amplitude mismatch at target_state={k}"
            )

    # ── Basis-state bras (MSB convention) ────────────────────────────────────

    def test_basis_bras_msb_convention(self):
        """_basis_bras must follow CUDA-Q MSB convention (q[0] = MSB)."""
        zero = np.array([1, 0], dtype=np.complex128)
        one = np.array([0, 1], dtype=np.complex128)
        # For 2 qubits, state 2 = binary '10' → q[0]=1, q[1]=0
        bras = _basis_bras(2, 2, zero, one)
        np.testing.assert_array_equal(bras[0], one)   # q[0] = 1
        np.testing.assert_array_equal(bras[1], zero)  # q[1] = 0
        # state 1 = binary '01' → q[0]=0, q[1]=1
        bras = _basis_bras(2, 1, zero, one)
        np.testing.assert_array_equal(bras[0], zero)
        np.testing.assert_array_equal(bras[1], one)

    # ── X gate flips state ────────────────────────────────────────────────────

    def test_ry_pi_gives_pure_one_state(self):
        """RY(π)|0⟩ = |1⟩; amplitude for |0⟩ = 0, amplitude for |1⟩ = 1."""
        @cudaq.kernel
        def circ(t: float):
            q = cudaq.qvector(1)
            ry(t, q[0])

        circuit = capture_circuit(circ, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)
        thetas = np.array([[math.pi]])
        rot = make_rotation_matrices(thetas, codes)

        amp0 = batched_contract(circuit, positions, rot, mode='amplitude', target_state=0)
        amp1 = batched_contract(circuit, positions, rot, mode='amplitude', target_state=1)
        np.testing.assert_allclose(np.abs(amp0[0]) ** 2, 0.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(amp1[0]) ** 2, 1.0, atol=1e-10)

    # ── Identity parameter sweep ──────────────────────────────────────────────

    def test_rx_probability_matches_formula(self):
        """
        For circuit |0⟩ → RX(θ) → measure: P(|0⟩) = cos²(θ/2).
        Verify for a sweep of θ values in a single batch call.
        """
        @cudaq.kernel
        def circ(t: float):
            q = cudaq.qvector(1)
            rx(t, q[0])

        circuit = capture_circuit(circ, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)

        theta_vals = np.linspace(0, 2 * math.pi, 20)
        thetas = theta_vals[:, np.newaxis]  # (20, 1)
        rot = make_rotation_matrices(thetas, codes)

        amps = batched_contract(circuit, positions, rot, mode='amplitude', target_state=0)
        probs = np.abs(amps) ** 2
        expected = np.cos(theta_vals / 2) ** 2
        np.testing.assert_allclose(probs, expected, atol=1e-8)

    # ── Multi-qubit: batch independence ──────────────────────────────────────

    def test_batch_items_are_independent(self):
        """Different parameter sets in the same batch must not interfere."""
        @cudaq.kernel
        def circ(t0: float, t1: float):
            q = cudaq.qvector(2)
            ry(t0, q[0])
            rz(t1, q[1])
            cx(q[0], q[1])

        circuit = capture_circuit(circ, 0.0, 0.0)
        positions, codes_list, _ = circuit.get_parametric_gate_info()
        codes = np.array(codes_list)

        rng = np.random.default_rng(99)
        B = 8
        thetas = rng.uniform(0, math.pi, (B, len(positions)))
        rot = make_rotation_matrices(thetas, codes)
        svs_batch = batched_contract(circuit, positions, rot, mode='statevector')
        svs_batch = svs_batch.reshape(B, -1)

        for i in range(B):
            circuit_i = capture_circuit(circ, thetas[i, 0], thetas[i, 1])
            sv_ref = circuit_i.get_state_vector()
            np.testing.assert_allclose(svs_batch[i], sv_ref, atol=1e-4,
                                       err_msg=f"Batch item {i} mismatch")


# ============================================================================
# 5. End-to-end: benchmark parity & QSVM
# ============================================================================

@pytest.mark.skipif(
    not (HAS_CUDAQ and HAS_TORCH),
    reason="Requires CUDA-Q and torch"
)
class TestEndToEnd:
    """Compare against the benchmark's ad-hoc reference implementation."""

    @staticmethod
    def _bench_make_rot_mats(thetas, codes):
        flat = thetas.to(torch.float64).reshape(-1)
        N = flat.numel()
        half = flat * 0.5
        cosv = torch.cos(half).to(torch.complex128)
        sinv = torch.sin(half).to(torch.complex128)
        phase = torch.exp(-1j * half)
        conjp = phase.conj()
        phaseP = torch.exp(1j * flat.to(torch.complex128))
        reps = N // codes.numel()
        cf = codes.to(torch.long).repeat(reps).to(flat.device)
        is_rx = cf == 0; is_ry = cf == 1; is_rz = cf == 2; is_p = cf == 3
        out = torch.empty((N, 2, 2), dtype=torch.complex128, device=flat.device)
        out[:, 0, 0] = torch.where(is_p, 1.0+0j, torch.where(is_rz, phase, cosv))
        out[:, 1, 1] = torch.where(is_p, phaseP, torch.where(is_rz, conjp, cosv))
        out[:, 0, 1] = torch.where(is_rx, -1j*sinv, torch.where(is_ry, -sinv, 0.0+0j))
        out[:, 1, 0] = torch.where(is_rx, -1j*sinv, torch.where(is_ry, sinv, 0.0+0j))
        return out.view(*thetas.shape, 2, 2)

    @staticmethod
    def _bench_build_args(circuit, param_positions, batch_rot_mats):
        batch_idx = circuit.max_index + 1
        param_set = set(param_positions)
        dev = batch_rot_mats.device
        args = []
        zero_ket = torch.tensor([1.0, 0.0], dtype=torch.complex128, device=dev)
        for idx in circuit.initial_indices:
            args += [zero_ket, [idx]]
        pc = 0
        for i, (t, idx) in enumerate(zip(circuit.gate_tensors, circuit.gate_indices)):
            if i in param_set:
                args += [batch_rot_mats[:, pc], [batch_idx] + list(idx)]
                pc += 1
            else:
                args += [torch.tensor(t, dtype=torch.complex128, device=dev), list(idx)]
        zero_bra = torch.tensor([1.0, 0.0], dtype=torch.complex128, device=dev)
        for idx in circuit.output_indices:
            args += [zero_bra, [idx]]
        args.append([batch_idx])
        return args

    def test_rotation_matrices_match_benchmark(self):
        """make_rotation_matrices must exactly match the benchmark reference."""
        rng = np.random.default_rng(42)
        B, P = 10, 12
        thetas_np = rng.standard_normal((B, P))
        codes_np = np.array([0, 1, 2, 3] * 3)
        ours = make_rotation_matrices(thetas_np, codes_np, backend='torch')
        theirs = self._bench_make_rot_mats(
            torch.tensor(thetas_np), torch.tensor(codes_np, dtype=torch.long)
        )
        np.testing.assert_allclose(
            ours.cpu().numpy(), theirs.cpu().numpy(), atol=1e-14
        )

    def test_qsvm_build_args_match_benchmark(self):
        """build_batched_args must produce the same amplitudes as benchmark."""
        n = 2

        @cudaq.kernel
        def qsvm(x: list[float], neg_y: list[float], n: int):
            q = cudaq.qvector(n)
            for i in range(n): h(q[i])
            for i in range(n): rz(x[i], q[i]); ry(x[i], q[i])
            for i in range(n - 1): cx(q[i], q[i + 1])
            for i in range(n): rz(x[i], q[i])
            for i in range(n): rz(neg_y[i], q[i])
            for i in range(n - 2, -1, -1): cx(q[i], q[i + 1])
            for i in range(n): ry(neg_y[i], q[i]); rz(neg_y[i], q[i])
            for i in range(n): h(q[i])

        dummy = [0.0] * n
        circuit = capture_circuit(qsvm, dummy, dummy, n)
        positions, codes_list, _ = circuit.get_parametric_gate_info(
            gate_names=['rx', 'ry', 'rz']
        )
        codes = np.array(codes_list)

        B = 7
        thetas_np = np.random.default_rng(3).standard_normal((B, len(positions)))
        rot = make_rotation_matrices(thetas_np, codes, backend='torch')

        our_args = build_batched_args(circuit, positions, rot, mode='amplitude')
        bench_args = self._bench_build_args(circuit, positions, rot)

        our_out = torch.einsum(*our_args)
        bench_out = torch.einsum(*bench_args)
        np.testing.assert_allclose(
            our_out.cpu().numpy(), bench_out.cpu().numpy(), atol=1e-12
        )

    def test_qsvm_diagonal_kernel_is_one(self):
        """
        K(x, x) = |⟨0|U†(x)U(x)|0⟩|² = 1.
        Verify for a batch of 5 different x values simultaneously.
        """
        n = 3

        @cudaq.kernel
        def qsvm(x: list[float], neg_y: list[float], n: int):
            q = cudaq.qvector(n)
            for i in range(n): h(q[i])
            for i in range(n): rz(x[i], q[i]); ry(x[i], q[i])
            for i in range(n - 1): cx(q[i], q[i + 1])
            for i in range(n): rz(x[i], q[i])
            for i in range(n): rz(neg_y[i], q[i])
            for i in range(n - 2, -1, -1): cx(q[i], q[i + 1])
            for i in range(n): ry(neg_y[i], q[i]); rz(neg_y[i], q[i])
            for i in range(n): h(q[i])

        dummy = [0.0] * n
        circuit = capture_circuit(qsvm, dummy, dummy, n)
        positions, codes_list, _ = circuit.get_parametric_gate_info(
            gate_names=['rx', 'ry', 'rz']
        )
        codes = np.array(codes_list)

        # Build thetas for K(x, x): look up the exact gate ordering from the circuit
        rng = np.random.default_rng(0)
        B = 5
        x_batch = rng.uniform(-1, 1, (B, n))  # (B, n)

        # For each sample, fill thetas according to gate ordering
        info_list = circuit.get_parametric_gate_info(gate_names=['rx', 'ry', 'rz'])[2]
        P = len(positions)
        thetas = np.zeros((B, P))
        for b in range(B):
            x = x_batch[b]
            neg_x = -x
            # Map each parametric gate to its angle: x-gates use x, neg_y-gates use -x
            # The first half of parametric gates belong to φ(x), the second to φ†(y)
            # We determine assignment by whether the original_param is from x or neg_y;
            # since dummy=0, all original_params are 0 — use index-based split instead.
            # Gate ordering from QSVM circuit (n=3):
            #   [0..2n-1]: rz/ry interleaved for x part
            #   [2n..3n-1]: rz for x part
            #   [3n..4n-1]: rz for -y part
            #   [4n..6n-1]: ry/rz interleaved for -y part
            # Simpler: assign each gate from its qubit and position in info_list
            for p_idx, inf in enumerate(info_list):
                qubit = inf.qubit
                pos = inf.position
                # x-gates are in the first half of the circuit (before the midpoint gate)
                mid_gate = len(circuit.gates) // 2
                if pos < mid_gate:
                    thetas[b, p_idx] = x[qubit]
                else:
                    thetas[b, p_idx] = neg_x[qubit]

        rot = make_rotation_matrices(thetas, codes)
        amps = batched_contract(circuit, positions, rot, mode='amplitude')
        kernel_vals = np.abs(amps) ** 2
        np.testing.assert_allclose(kernel_vals, 1.0, atol=1e-4,
                                   err_msg="K(x,x) != 1 for some batch item")

    def test_qsvm_off_diagonal_symmetric(self):
        """
        K(x, y) = K(y, x) for distinct x, y.
        Build K(x, y) and K(y, x) in the same batch and compare.
        """
        n = 2

        @cudaq.kernel
        def qsvm(x: list[float], neg_y: list[float], n: int):
            q = cudaq.qvector(n)
            for i in range(n): h(q[i])
            for i in range(n): rz(x[i], q[i]); ry(x[i], q[i])
            for i in range(n - 1): cx(q[i], q[i + 1])
            for i in range(n): rz(x[i], q[i])
            for i in range(n): rz(neg_y[i], q[i])
            for i in range(n - 2, -1, -1): cx(q[i], q[i + 1])
            for i in range(n): ry(neg_y[i], q[i]); rz(neg_y[i], q[i])
            for i in range(n): h(q[i])

        dummy = [0.0] * n
        circuit = capture_circuit(qsvm, dummy, dummy, n)
        positions, codes_list, info_list = circuit.get_parametric_gate_info(
            gate_names=['rx', 'ry', 'rz']
        )
        codes = np.array(codes_list)
        P = len(positions)

        rng = np.random.default_rng(7)
        x_val = rng.uniform(-1, 1, n)
        y_val = rng.uniform(-1, 1, n)

        def build_thetas(x, neg_y):
            th = np.zeros(P)
            mid = len(circuit.gates) // 2
            for p_idx, inf in enumerate(info_list):
                if inf.position < mid:
                    th[p_idx] = x[inf.qubit]
                else:
                    th[p_idx] = neg_y[inf.qubit]
            return th

        thetas_xy = build_thetas(x_val, -y_val)
        thetas_yx = build_thetas(y_val, -x_val)
        thetas_batch = np.stack([thetas_xy, thetas_yx])  # (2, P)

        rot = make_rotation_matrices(thetas_batch, codes)
        amps = batched_contract(circuit, positions, rot, mode='amplitude')
        K_xy = np.abs(amps[0]) ** 2
        K_yx = np.abs(amps[1]) ** 2
        np.testing.assert_allclose(K_xy, K_yx, atol=1e-4,
                                   err_msg="QSVM kernel not symmetric: K(x,y) != K(y,x)")


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
