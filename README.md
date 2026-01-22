# Einsum NVQIR Simulator

A CUDA-Q NVQIR simulator that intercepts quantum gates and builds Einsum expressions for tensor network contraction.

## Project Structure

```
einsum_nvqir/
├── src/
│   └── EinsumSimulator.cpp    # Main simulator implementation
├── tests/
│   └── test_einsum.py         # Python test script
├── scripts/
│   ├── docker-build.sh        # Build script using Docker
│   ├── docker-test.sh         # Test script using Docker
│   └── docker-shell.sh        # Interactive Docker shell
├── build/                     # Build output directory
├── CMakeLists.txt             # CMake configuration
└── README.md
```

## Requirements

- Docker with `nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest` image
- (Optional) CLion IDE for development

## Quick Start

### 1. Build the simulator

```bash
./scripts/docker-build.sh
```

This will:
- Start a Docker container with CUDA-Q
- Install build dependencies (cmake, g++)
- Compile `libnvqir-einsum.so`

### 2. Test the simulator

```bash
./scripts/docker-test.sh
```

This will:
- Copy the library to CUDA-Q's lib directory
- Create the target configuration
- Run Python tests

### 3. Interactive development

```bash
./scripts/docker-shell.sh
```

This opens an interactive shell in the Docker container.

## Usage in Python

```python
import cudaq

# Set the Einsum simulator as target
cudaq.set_target("einsum")

@cudaq.kernel
def my_circuit():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

# Run the circuit - this will print Einsum index tracking
results = cudaq.sample(my_circuit)
```

## CLion Development Setup

### Option 1: Remote Docker Toolchain (Recommended)

1. Open CLion Settings → Build, Execution, Deployment → Toolchains
2. Add a new "Docker" toolchain
3. Set the image to `nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest`
4. Configure CMake options: `-DCUDAQ_ROOT=/opt/nvidia/cudaq`

### Option 2: Use Docker scripts directly

CLion can run the build scripts as external tools:
1. Settings → Tools → External Tools → Add
2. Configure `docker-build.sh` as a build tool
3. Configure `docker-test.sh` as a run tool

### Header Files for Code Completion

To enable code completion in CLion without Docker:

1. Extract headers from the Docker image:
```bash
docker run --rm -v $(pwd):/out nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest \
    bash -c "cp -r /opt/nvidia/cudaq/include /out/cudaq-include"
```

2. Add to CMakeLists.txt:
```cmake
if(EXISTS "${CMAKE_SOURCE_DIR}/cudaq-include")
    include_directories(${CMAKE_SOURCE_DIR}/cudaq-include)
endif()
```

## API Overview

The `EinsumBuilder` class inherits from `nvqir::CircuitSimulatorBase<double>` and implements:

| Method | Description |
|--------|-------------|
| `addQubitToState()` | Allocate and track a new qubit index |
| `applyGate(task)` | Intercept gate operations and record indices |
| `measureQubit(idx)` | Handle qubit measurement |
| `sample(qubits, shots)` | Return sampling results |

## Next Steps for Development

1. **Generate proper Einsum strings**: Convert gate records to standard Einsum notation
2. **Store gate matrices**: Compute and store rotation matrices with resolved parameters
3. **cuTensorNet integration**: Build `cutensornetNetworkDescriptor` directly
4. **Export functionality**: Add methods to export the tensor network graph

## References

- [CUDA-Q Documentation](https://nvidia.github.io/cuda-quantum/)
- [NVQIR Architecture](https://nvidia.github.io/cuda-quantum/latest/using/backends.html)
- [cuTensorNet](https://docs.nvidia.com/cuda/cutensor/)
