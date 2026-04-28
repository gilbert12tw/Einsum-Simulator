## qsvm_cutensor_backend

Pybind11 C++ backend for QSVM tensor-network contraction with cuTENSOR.
It is used by `qsvm_cudaq_cpp_backend.py` to validate and time the optional
C++ multi-stream contraction path in `04_cutn-qsvm_cudaq.ipynb`.

### Prerequisites

Set these variables if CUDA or cuTENSOR are not installed in default locations:

```bash
export CUDA_ROOT=/path/to/cuda-12.5
export CUTENSOR_ROOT=/path/to/libcutensor-linux-x86_64-archive
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUTENSOR_ROOT/lib/12:$LD_LIBRARY_PATH
```

For tcsh:

```tcsh
setenv CUDA_ROOT /path/to/cuda-12.5
setenv CUTENSOR_ROOT /path/to/libcutensor-linux-x86_64-archive
setenv LD_LIBRARY_PATH "$CUDA_ROOT/lib64:$CUTENSOR_ROOT/lib/12:$LD_LIBRARY_PATH"
```

`CUTENSOR_ROOT` should contain `include/cutensor.h` and `lib/12/libcutensor.so`.
`LD_LIBRARY_PATH` helps the Python extension and standalone binaries find CUDA
and cuTENSOR runtime libraries on systems where rpath is insufficient.

### Build From This Directory

```bash
bash build_backend.sh
```

If your home directory is space-limited, point pip temporary/cache directories to
scratch before running the build script.

### Quick Check

```bash
python3 -c "import qsvm_cutensor_backend as m; print(m.version())"
```

Expected version for this bundle:

```text
0.10.0-cutensor
```

### Python Entry Points

The pybind module exports:

- `init_backend(config)`: initialize tensor modes, contraction path, and optional stream paths.
- `contract_batch_complex_from_torch_ptr_table(ptr_table)`: run contractions from CUDA tensor pointers and return complex amplitudes.
- `contract_batch_from_torch_ptr_table(ptr_table)`: legacy helper returning `|amplitude|^2`.

Most users should not call these directly; use `qsvm_cudaq_cpp_backend.py` from
the bundle root instead.
