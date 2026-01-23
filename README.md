# Einsum NVQIR Simulator

Convert CUDA-Q quantum circuits to einsum tensor network representation for use with `torch.einsum` or `opt_einsum`.

## Features

- Intercept CUDA-Q gate operations and capture full gate matrices
- Integer index system
- Output PyTorch sublist format for large circuits

## Installation

Requires CUDA-Q environment. Build the simulator:

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### 1. Capture Circuit

```python
from cudaq_einsum import capture_circuit

@cudaq.kernel
def ghz(n: int):
    qubits = cudaq.qvector(n)
    h(qubits[0])
    for i in range(n - 1):
        cx(qubits[i], qubits[i + 1])

# Execute kernel and capture circuit structure
circuit = capture_circuit(ghz, 3)
```

### 2. Convert to Einsum and Execute

```python
import torch

# Get torch.einsum sublist format arguments
args = circuit.to_torch_sublist_args()
state = torch.einsum(*args)

print(state)  # GHZ state: [0.707, 0, 0, 0, 0, 0, 0, 0.707]
```

### 3. Or Use opt_einsum

```python
import opt_einsum

tensors, indices, output = circuit.to_operands_and_subscripts()
# Compose your own opt_einsum call
```

## Why Integer Indices?

Traditional einsum string format `"ij,jk->ik"` only supports character indices, which is insufficient for large circuits.

This project uses PyTorch sublist format:
```python
# String format 
torch.einsum("ij,jk->ik", A, B)

# Sublist format 
torch.einsum(A, [0, 1], B, [1, 2], [0, 2])
```

## Status

- [x] C++ Sidecar API
- [x] Gate matrix capture
- [x] JSON serialization
- [ ] Python wrapper
- [ ] End-to-end tests

## License

MIT
