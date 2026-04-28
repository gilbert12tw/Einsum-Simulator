"""
Adapter from captured CUDA-Q circuits to the optional C++ multi-stream backend.
"""

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .multistream_tree import ContractionTree
    from .qsvm_contract_utils import extract_multistream_parts
    from .qsvm_cpp_backend import _import_cpp_module, init_cpp_backend
except ImportError:
    from multistream_tree import ContractionTree
    from qsvm_contract_utils import extract_multistream_parts
    from qsvm_cpp_backend import _import_cpp_module, init_cpp_backend


def _idx_to_symbol(idx: int) -> str:
    return chr(0xE000 + int(idx))


def _subscript_from_indices(indices) -> str:
    return "".join(_idx_to_symbol(i) for i in indices)


def _flatten_triple_path(path) -> List[int]:
    flat = []
    for a, b, c in path:
        flat.extend([int(a), int(b), int(c)])
    return flat


def _basis_bras(num_qubits, state_index, zero_ket, one_ket):
    bras = []
    for i in range(num_qubits):
        bit = (state_index >> (num_qubits - 1 - i)) & 1
        bras.append(one_ket if bit else zero_ket)
    return bras


def _build_sample_operands(
    circuit, param_positions, batch_rot_mats, sample_idx, target_state=0
):
    import torch

    if not torch.is_tensor(batch_rot_mats):
        batch_rot_mats = torch.as_tensor(batch_rot_mats)
    if not batch_rot_mats.is_cuda:
        raise RuntimeError("C++ backend requires CUDA torch tensors for batch_rot_mats.")

    device = batch_rot_mats.device
    zero_ket = torch.tensor([1.0, 0.0], dtype=torch.complex128, device=device)
    one_ket = torch.tensor([0.0, 1.0], dtype=torch.complex128, device=device)

    param_set = set(param_positions)
    operands = []
    subscripts = []

    for idx in circuit.initial_indices:
        operands.append(zero_ket)
        subscripts.append(_subscript_from_indices([idx]))

    param_count = 0
    for gate_index, (tensor, indices) in enumerate(
        zip(circuit.gate_tensors, circuit.gate_indices)
    ):
        if gate_index in param_set:
            operands.append(batch_rot_mats[sample_idx, param_count].contiguous())
            param_count += 1
        else:
            tensor = np.asarray(tensor, dtype=np.complex128)
            operands.append(
                torch.as_tensor(tensor, dtype=torch.complex128, device=device).contiguous()
            )
        subscripts.append(_subscript_from_indices(indices))

    for idx, bra in zip(
        circuit.output_indices,
        _basis_bras(circuit.num_qubits, target_state, zero_ket, one_ket),
    ):
        operands.append(bra)
        subscripts.append(_subscript_from_indices([idx]))

    return operands, subscripts


def _build_contract_config(
    circuit, param_positions, batch_rot_mats, target_state=0, verbose=False
):
    from cuquantum import Network, NetworkOptions

    operands, input_subscripts = _build_sample_operands(
        circuit, param_positions, batch_rot_mats, sample_idx=0, target_state=target_state
    )
    expr = ",".join(input_subscripts) + "->"
    network = Network(expr, *operands, options=NetworkOptions(blocking="auto"))
    path, _ = network.contract_path()
    network.free()

    tree = ContractionTree(expr, path)
    full_path = tree.postorder_traverse_contractions()
    stream_0, stream_1, stream_2, _, _ = extract_multistream_parts(
        tree, full_path, verbose=verbose
    )

    return {
        "input_subscripts": input_subscripts,
        "triple_path": _flatten_triple_path(full_path),
        "stream_0": _flatten_triple_path(stream_0),
        "stream_1": _flatten_triple_path(stream_1),
        "stream_2": _flatten_triple_path(stream_2),
    }


def _to_torch_ptr_table(opers):
    import torch

    if len(opers) == 0:
        return np.empty((0, 0), dtype=np.uint64)
    n_operands = len(opers[0])
    ptrs = np.empty((len(opers), n_operands), dtype=np.uint64)
    for sample_index, sample in enumerate(opers):
        if len(sample) != n_operands:
            raise ValueError("Inconsistent operand count across samples.")
        for tensor_index, tensor in enumerate(sample):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("C++ backend requires torch.Tensor operands.")
            if not tensor.is_cuda:
                raise TypeError("C++ backend requires CUDA tensors.")
            if not tensor.is_contiguous():
                raise TypeError("C++ backend requires contiguous tensors.")
            if tensor.dtype != torch.complex128:
                raise TypeError("C++ backend requires torch.complex128 tensors.")
            ptrs[sample_index, tensor_index] = tensor.data_ptr()
    return ptrs


def prepare_cpp_backend(
    circuit,
    param_positions,
    batch_rot_mats,
    target_state: Optional[int] = None,
    module_name: str = "qsvm_cutensor_backend",
    verbose: bool = False,
) -> Dict[str, Any]:
    if target_state is None:
        target_state = 0

    config = _build_contract_config(
        circuit,
        param_positions,
        batch_rot_mats,
        target_state=target_state,
        verbose=verbose,
    )
    init_cpp_backend(config, module_name=module_name)

    batch_size = int(batch_rot_mats.shape[0])
    opers = [
        _build_sample_operands(
            circuit, param_positions, batch_rot_mats, sample_idx, target_state=target_state
        )[0]
        for sample_idx in range(batch_size)
    ]
    ptr_table = _to_torch_ptr_table(opers)
    cpp_mod = _import_cpp_module(module_name)
    if not hasattr(cpp_mod, "contract_batch_complex_from_torch_ptr_table"):
        raise RuntimeError(
            f"{module_name} must provide contract_batch_complex_from_torch_ptr_table. "
            "Please rebuild cpp_backend."
        )

    return {
        "config": config,
        "opers": opers,
        "ptr_table": ptr_table,
        "cpp_mod": cpp_mod,
        "device": batch_rot_mats.device,
        "batch_size": batch_size,
    }


def run_cpp_backend(prepared):
    import torch

    cpp_mod = prepared["cpp_mod"]
    ptr_table = prepared["ptr_table"]
    device = prepared["device"]
    result_np = np.asarray(
        cpp_mod.contract_batch_complex_from_torch_ptr_table(ptr_table),
        dtype=np.complex128,
    )
    return torch.from_numpy(result_np).to(device=device)
