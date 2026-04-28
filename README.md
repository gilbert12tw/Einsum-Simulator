# Einsum NVQIR Simulator

Convert CUDA-Q quantum circuits to einsum tensor network representations for contraction with `cuQuantum`, `opt_einsum`, or any other backend.

The simulator intercepts gate operations from CUDA-Q kernels and records them as a tensor network, without performing any actual quantum simulation. The resulting circuit can be contracted by any einsum backend.

## Structure and Benchmarks
Note: The `benchmarks/` folder in the repository contains cutting-edge research and applications. These are **not included** in the distributed Python package. Future updates will add more applications and benchmarks to test correctness and timing.

## Features

- Intercept CUDA-Q gate operations and capture full gate matrices
- Integer index system — no alphabet limit; real circuits exceed 52 unique indices
- Output in interleaved (sublist) format compatible with cuQuantum `Network`, `opt_einsum`, and PyTorch
- Works with `cuquantum.tensornet.Network`, `opt_einsum`, `torch.einsum`, and other contraction backends
- **Batched evaluation** — evaluate B parameter sets in a single einsum contraction; see [Batched evaluation](#batched-evaluation)

## Prerequisites

This project requires a **CUDA-Q** environment with the NVQIR runtime. Choose one of the following installation methods:

- **conda** (recommended for local development): [CUDA-Q conda instructions](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)
- **pip**: `pip install cuda-quantum`
- **Docker**: see [Docker development](#docker-development) below

---

## Installation (For Users)

### pip install (recommended)

Requires CMake and a C++ compiler (`g++`) in addition to CUDA-Q.

```bash
# 1. Activate your CUDA-Q environment
conda activate cudaq-env          # conda
# or: source /path/to/cudaq/set_env.sh  # pip / manual

# 2. Build and install the package
pip install .

# 3. Register the simulator with CUDA-Q (one-time step)
cudaq-einsum-install
# Or via python:
# python -c "import cudaq_einsum; cudaq_einsum.install_cudaq_target()"
```

Optional dependencies:

```bash
pip install ".[torch]"       # add PyTorch support
pip install ".[opt_einsum]"  # add opt_einsum support
pip install ".[all]"         # add all optional deps
```

> **Note:** `cudaq-einsum-install` copies `libnvqir-einsum.so` and `einsum.yml`
> into your CUDA-Q installation directory. You need write access to that directory.
> Re-run it whenever you reinstall the package.

---

## Development & Contributing

The repository uses a standard standard `src` layout for the Python package (`src/cudaq_einsum`) and a `cpp/` directory for the C++ library.

### Automated Testing

For ease of development, the repository provides unified scripts that auto-detect your CUDA-Q installation, build the library using pip, and run the complete test suite.

**Option 1: Local Environment (Conda or Pip)**
Ensure your `conda` or local environment with CUDA-Q is active, then run:

```bash
./scripts/local-build.sh
```

**Option 2: Docker Environment**
If you don't want to pollute your host system, you can use the pre-configured CUDA-Q nightly Docker container. The script will automatically build the development image (if it doesn't exist), mount your local repository, build the library, and run tests inside a disposable container setup:

```bash
./scripts/docker-test.sh
```

For an interactive development shell inside Docker:

```bash
./scripts/docker-shell.sh
# Inside the container, you can manually trigger build and tests:
./scripts/build-and-test.sh
```

---

## Troubleshooting & Common Pitfalls

- **`RuntimeError: Cannot find CUDA-Q installation` during `pip install .`**
  Pip isolated builds sometimes block access to your host system's `sys.path`. The `setup.py` contains explicit fallback heuristics (e.g. searching conda prefix and standard Python paths). If auto-detection fails, explicitly point to your CUDA-Q root installation path:
  ```bash
  export CUDAQ_ROOT=/path/to/your/cuda-quantum-installation
  pip install .
  ```

- **`ERROR: Library not found: /path/to/libnvqir-einsum.so`**
  This implies the python bindings loaded but the bundled C++ `.so` file was not created successfully alongside it. Rebuild the package while enforcing recompilation:
  ```bash
  pip install . --force-reinstall --no-cache-dir
  ```

- **Runtime `get_einsum_length` crashes or memory corruptions**
  This happens if your user application links to a DIFFERENT instance of the CUDA-Q libraries than the `libnvqir-einsum.so` bindings. Ensure you run your code under the exact same conda environment or Docker image that you packaged it with.

---

## Usage

### 1. Capture a circuit

```python
import cudaq
from cudaq_einsum import capture_circuit

@cudaq.kernel
def ghz(n: int):
    qubits = cudaq.qvector(n)
    h(qubits[0])
    for i in range(n - 1):
        cx(qubits[i], qubits[i + 1])

# Execute kernel and capture circuit structure
circuit = capture_circuit(ghz, 3)

print(f"Qubits: {circuit.num_qubits}, Gates: {len(circuit.gates)}")
```

### 2. Contract with cuQuantum (recommended)

```python
from cuquantum.tensornet import Network, NetworkOptions

args = circuit.to_torch_sublist_args()  # interleaved format
options = NetworkOptions(blocking="auto", device_id=0)
network = Network(*args, options=options)
_, _ = network.contract_path(optimize={'slicing': {'min_slices': 32}})
state = network.contract()
network.free()
state_vector = state.reshape(-1)
```

### 3. Contract with opt_einsum

```python
import opt_einsum

args = circuit.to_torch_sublist_args()
state = opt_einsum.contract(*args, backend='torch')
state_vector = state.reshape(-1)
```

### 4. Get the state vector directly (small circuits only)

```python
# Uses NumPy einsum internally — only practical for circuits with ≤ ~15 qubits.
state_vector = circuit.get_state_vector()
print(state_vector)
```

## Batched evaluation

CUDA-Q cannot natively evaluate parameterized circuits for multiple parameter
sets simultaneously.  The batched API exploits the tensor-network
representation: for each parametric gate, a *(B, 2, 2)* batch of rotation
matrices replaces the usual *(2, 2)* tensor, and a single einsum contraction
returns *B* amplitudes at once.  This avoids re-running the CUDA-Q kernel
per parameter set and is directly compatible with `cuQuantum`'s `Network` API.

### Quick start

```python
import numpy as np
import cudaq
from cudaq_einsum import (
    capture_circuit, make_rotation_matrices,
    build_batched_args, batched_contract,
)

@cudaq.kernel
def feature_map(x: list[float], neg_y: list[float], n: int):
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
        rz(x[i], q[i])
        ry(x[i], q[i])
    for i in range(n - 1):
        cx(q[i], q[i + 1])
    for i in range(n):
        rz(neg_y[i], q[i])
        ry(neg_y[i], q[i])
    for i in range(n):
        h(q[i])

n = 4
# 1. Capture once with dummy parameters
circuit = capture_circuit(feature_map, [0.0]*n, [0.0]*n, n)

# 2. Identify parametric gates (returns positions + gate-type codes)
positions, codes_list, info_list = circuit.get_parametric_gate_info(
    gate_names=['rx', 'ry', 'rz']
)
codes = np.array(codes_list)   # (P,) int array: rx=0, ry=1, rz=2, p/r1=3

# 3. Build (B, P) angle matrix — one row per parameter set
B = 64
thetas = np.random.uniform(-np.pi, np.pi, (B, len(positions)))

# 4. Build (B, P, 2, 2) complex128 rotation matrices
rot_mats = make_rotation_matrices(thetas, codes, backend='numpy')
# backend='torch' for GPU / autograd support

# 5. Single batched contraction → (B,) complex amplitudes
amplitudes = batched_contract(circuit, positions, rot_mats, mode='amplitude')
kernel_values = np.abs(amplitudes) ** 2   # (B,) real

# For large circuits: use cuQuantum Network directly
# args = build_batched_args(circuit, positions, rot_mats, mode='amplitude')
# network = Network(*args, options=NetworkOptions(device_id=0))
# amplitudes = network.contract()
```

Both `mode='amplitude'` (returns a scalar ⟨target|ψ⟩ per batch item) and
`mode='statevector'` (returns the full *(B, 2, 2, …, 2)* state tensor) are
supported.

A complete, runnable QSVM example is in [`examples/11_qsvm_batching.py`](examples/11_qsvm_batching.py).

### Batching API reference

| Function / Method | Description |
|---|---|
| `make_rotation_matrices(thetas, codes, backend, device)` | Build `(B, P, 2, 2)` rotation matrices for a batch of parameter sets |
| `build_batched_args(circuit, positions, rot_mats, mode, target_state)` | Build sublist einsum args with batch dimension |
| `batched_contract(circuit, positions, rot_mats, mode, target_state)` | Convenience wrapper: build args + contract |
| `EinsumCircuit.get_parametric_gate_info(gate_names)` | Identify parametric gates; returns `(positions, codes, info_list)` |

**Gate-type codes**: `rx=0`, `ry=1`, `rz=2`, `r1/p=3`

**Modes**:
- `'amplitude'` — project onto a basis state; output `(B,)` complex. Default target state is `|0...0⟩`.
- `'statevector'` — full output tensor `(B, 2, 2, ..., 2)`.

> **Note:** `numpy.einsum` hangs on deep circuits (>~10 gates).
> `batched_contract` automatically uses `opt_einsum` when available.
> For production, pass the output of `build_batched_args()` directly to
> `cuQuantum`'s `Network` for GPU acceleration and optimal path finding.

## Experimental multi-stream backend

An optional C++/cuTENSOR backend can be used for QSVM-style batched amplitude
evaluation after the pybind module has been built in `cpp_backend/`.

```python
from cudaq_einsum.multistream import multistream_batched_contract

amps = multistream_batched_contract(
    captured_circuit,
    param_positions,
    rotation_matrices,
)
```

If you need to split setup time from contraction time, use:

```python
from cudaq_einsum.multistream import (
    prepare_multistream_backend,
    run_prepared_multistream,
)
```

## API reference

| Function / Method | Description |
|---|---|
| `capture_circuit(kernel, *args)` | Run kernel, return `EinsumCircuit` |
| `capture_circuit_json(kernel, *args)` | Return raw circuit JSON string |
| `install_cudaq_target(cudaq_root=None)` | Copy `.so` + `.yml` to CUDA-Q dirs |
| `EinsumCircuit.to_torch_sublist_args()` | PyTorch sublist format args |
| `EinsumCircuit.to_einsum_sublist_args()` | Integer index format (operands, subscripts) |
| `EinsumCircuit.to_einsum_args()` | NumPy string format |
| `EinsumCircuit.get_state_vector()` | Contracted state as 1D array (small circuits only) |
| `EinsumCircuit.contract()` | Contracted state as N-D tensor |
| `EinsumCircuit.get_parametric_gate_info(gate_names)` | Identify parametric gates for batching |

## Suppressing log output

By default the simulator runs silently. Gate-level logs (`[Einsum] Gate: ...`) are
disabled unless you set the environment variable:

```bash
EINSUM_VERBOSE=1 python your_script.py
```

## License

MIT
