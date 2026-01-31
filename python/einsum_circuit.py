"""
EinsumCircuit data class for representing quantum circuits as tensor networks.

This module provides the core data structure for holding circuit information
extracted from the C++ simulator, along with methods for building proper
tensor representations including controlled gates.
"""

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np


@dataclass
class GateInfo:
    """Information about a single gate operation."""
    name: str
    params: List[float]
    controls: List[int]
    targets: List[int]
    input_indices: List[int]
    output_indices: List[int]
    matrix: np.ndarray  # Raw matrix from C++ (may be 2x2 for controlled gates)


@dataclass
class EinsumCircuit:
    """
    Represents a quantum circuit in tensor network form.

    This class holds the complete circuit structure with properly built
    gate tensors suitable for einsum contraction.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        max_index: Maximum tensor index used.
        initial_indices: Initial tensor indices for each qubit.
        output_indices: Final tensor indices for each qubit.
        gates: List of GateInfo objects with gate details.
        gate_tensors: List of numpy arrays representing gate tensors.
        gate_indices: List of index tuples for each gate tensor.
    """
    num_qubits: int
    max_index: int
    initial_indices: List[int]
    output_indices: List[int]
    gates: List[GateInfo] = field(default_factory=list)
    gate_tensors: List[np.ndarray] = field(default_factory=list)
    gate_indices: List[Tuple[int, ...]] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_str: str) -> 'EinsumCircuit':
        """
        Parse JSON from the C++ sidecar and build an EinsumCircuit.

        Args:
            json_str: JSON string from the sidecar buffer.

        Returns:
            EinsumCircuit instance with properly constructed gate tensors.
        """
        data = json.loads(json_str)

        num_qubits = data['numQubits']
        max_index = data['maxIndex']
        initial_indices = [s['index'] for s in data['initialStates']]
        output_indices = data['outputIndices']

        gates = []
        gate_tensors = []
        gate_indices = []

        for g in data['gates']:
            # Parse matrix from [real, imag] pairs
            matrix_data = g['matrix']
            matrix_flat = np.array(
                [complex(elem[0], elem[1]) for elem in matrix_data],
                dtype=np.complex128
            )

            gate_info = GateInfo(
                name=g['name'],
                params=g['params'],
                controls=g['controls'],
                targets=g['targets'],
                input_indices=g['inputIndices'],
                output_indices=g['outputIndices'],
                matrix=matrix_flat
            )
            gates.append(gate_info)

            # Build proper tensor for this gate
            tensor = _build_gate_tensor(
                matrix_flat,
                num_controls=len(g['controls']),
                num_targets=len(g['targets'])
            )
            gate_tensors.append(tensor)

            # Build index tuple: (out_0, out_1, ..., in_0, in_1, ...)
            # Tensor shape is (2, 2, ..., 2) with 2*(num_controls + num_targets) dimensions
            # First half are output indices, second half are input indices
            indices = tuple(g['outputIndices']) + tuple(g['inputIndices'])
            gate_indices.append(indices)

        return cls(
            num_qubits=num_qubits,
            max_index=max_index,
            initial_indices=initial_indices,
            output_indices=output_indices,
            gates=gates,
            gate_tensors=gate_tensors,
            gate_indices=gate_indices
        )

    def get_initial_state_tensors(self) -> List[Tuple[np.ndarray, Tuple[int, ...]]]:
        """
        Get the initial |0> state tensors for all qubits.

        Returns:
            List of (tensor, indices) tuples for initial states.
        """
        zero_state = np.array([1.0, 0.0], dtype=np.complex128)
        return [(zero_state, (idx,)) for idx in self.initial_indices]

    def to_einsum_args(self) -> Tuple[str, List[np.ndarray]]:
        """
        Generate numpy.einsum arguments (subscripts string format).

        Returns:
            Tuple of (subscripts_string, operands_list).

        Note:
            This format uses single characters for indices, so it's limited
            to 52 indices (a-z, A-Z). For larger circuits, use sublist format.
        """
        if self.max_index > 52:
            raise ValueError(
                f"Circuit uses {self.max_index} indices, exceeding the 52 character limit. "
                "Use to_einsum_sublist_args() instead."
            )

        def idx_to_char(idx: int) -> str:
            if idx < 26:
                return chr(ord('a') + idx)
            else:
                return chr(ord('A') + idx - 26)

        operands = []
        subscript_parts = []

        # Initial states
        for state_tensor, (idx,) in self.get_initial_state_tensors():
            operands.append(state_tensor)
            subscript_parts.append(idx_to_char(idx))

        # Gate tensors
        for tensor, indices in zip(self.gate_tensors, self.gate_indices):
            operands.append(tensor)
            subscript_parts.append(''.join(idx_to_char(i) for i in indices))

        # Build subscripts string
        input_subscripts = ','.join(subscript_parts)
        output_subscripts = ''.join(idx_to_char(i) for i in self.output_indices)
        subscripts = f"{input_subscripts}->{output_subscripts}"

        return subscripts, operands

    def to_einsum_sublist_args(self) -> List[Any]:
        """
        Generate einsum arguments in sublist (interleaved) format.

        This format works with both numpy.einsum and torch.einsum,
        and supports arbitrary numbers of indices.

        Returns:
            List in format: [op1, [idx1a, idx1b, ...], op2, [idx2a, ...], ..., output_indices]

        Usage:
            args = circuit.to_einsum_sublist_args()
            result = np.einsum(*args)  # or torch.einsum(*args)
        """
        args = []

        # Initial states
        for state_tensor, indices in self.get_initial_state_tensors():
            args.append(state_tensor)
            args.append(list(indices))

        # Gate tensors
        for tensor, indices in zip(self.gate_tensors, self.gate_indices):
            args.append(tensor)
            args.append(list(indices))

        # Output indices
        args.append(list(self.output_indices))

        return args

    def to_torch_sublist_args(self) -> List[Any]:
        """
        Generate torch.einsum arguments in sublist format with torch tensors.

        Returns:
            List suitable for torch.einsum(*args).

        Raises:
            ImportError: If torch is not available.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_torch_sublist_args()")

        args = []

        # Initial states
        for state_tensor, indices in self.get_initial_state_tensors():
            args.append(torch.from_numpy(state_tensor))
            args.append(list(indices))

        # Gate tensors
        for tensor, indices in zip(self.gate_tensors, self.gate_indices):
            args.append(torch.from_numpy(tensor))
            args.append(list(indices))

        # Output indices
        args.append(list(self.output_indices))

        return args

    def contract(self) -> np.ndarray:
        """
        Contract the tensor network to compute the final state vector.

        Returns:
            Complex numpy array representing the final quantum state.
        """
        args = self.to_einsum_sublist_args()
        result = np.einsum(*args)
        return result

    def get_state_vector(self) -> np.ndarray:
        """
        Compute and return the flattened state vector.

        Returns:
            1D complex array of length 2^num_qubits.
        """
        return self.contract().flatten()

    def __repr__(self) -> str:
        gate_summary = ', '.join(g.name for g in self.gates[:5])
        if len(self.gates) > 5:
            gate_summary += f", ... ({len(self.gates)} total)"
        return (
            f"EinsumCircuit(num_qubits={self.num_qubits}, "
            f"gates=[{gate_summary}], "
            f"max_index={self.max_index})"
        )


def _build_gate_tensor(
    base_matrix: np.ndarray,
    num_controls: int,
    num_targets: int
) -> np.ndarray:
    """
    Build the full gate tensor from the base matrix.

    For controlled gates, the C++ side only provides the target gate matrix.
    This function reconstructs the full controlled unitary.

    Args:
        base_matrix: Flat array of the base gate matrix.
        num_controls: Number of control qubits.
        num_targets: Number of target qubits.

    Returns:
        Tensor of shape (2, 2, ..., 2) with 2*(num_controls + num_targets) dims.
        The first half of dimensions are output indices, second half are input.
    """
    total_qubits = num_controls + num_targets

    if num_controls == 0:
        # Simple single/multi-qubit gate - just reshape
        target_dim = 2 ** num_targets
        matrix = base_matrix.reshape(target_dim, target_dim)
        # Reshape to tensor form: (out_0, out_1, ..., in_0, in_1, ...)
        shape = (2,) * (2 * num_targets)
        # Matrix is in row-major order: M[out, in]
        # We need tensor with indices: T[out_0, out_1, ..., in_0, in_1, ...]
        return matrix.reshape(shape)

    # Controlled gate: |0><0| ⊗ I + |1><1| ⊗ ... ⊗ |1><1| ⊗ U
    # Build the full 2^n × 2^n matrix, then reshape to tensor
    full_dim = 2 ** total_qubits
    target_dim = 2 ** num_targets

    # Start with identity
    full_matrix = np.eye(full_dim, dtype=np.complex128)

    # The unitary U is applied to the target qubits only when ALL control qubits are |1>
    # This corresponds to the bottom-right block of the full matrix
    target_matrix = base_matrix.reshape(target_dim, target_dim)

    # Block start is at index where all controls are 1
    # If we have c controls and t targets, the block indices are:
    # [2^t * (2^c - 1), 2^t * 2^c - 1] = [full_dim - target_dim, full_dim - 1]
    block_start = full_dim - target_dim
    full_matrix[block_start:, block_start:] = target_matrix

    # Reshape to tensor form
    shape = (2,) * (2 * total_qubits)
    return full_matrix.reshape(shape)


def _matrix_to_tensor(matrix: np.ndarray, num_qubits: int) -> np.ndarray:
    """
    Convert a 2^n × 2^n matrix to tensor form.

    Args:
        matrix: Square matrix of dimension 2^n.
        num_qubits: Number of qubits (n).

    Returns:
        Tensor of shape (2, 2, ..., 2) with 2n dimensions.
    """
    dim = 2 ** num_qubits
    assert matrix.shape == (dim, dim), f"Expected {dim}x{dim} matrix"
    return matrix.reshape((2,) * (2 * num_qubits))
