#!/bin/bash
# Build, install, and test einsum simulator (run inside Docker container)
# All steps must be done in the SAME Docker session!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CUDAQ_LIB="/opt/nvidia/cudaq/lib"
CUDAQ_TARGETS="/opt/nvidia/cudaq/targets"

echo "============================================================"
echo "  Einsum NVQIR - Build, Install, and Test"
echo "============================================================"
echo ""

# Step 1: Build
echo "[1/4] Building C++ library..."
cd "$PROJECT_DIR"
mkdir -p build
cd build
cmake .. -DCUDAQ_ROOT=/opt/nvidia/cudaq > /dev/null
make -j$(nproc) 2>&1 | tail -3
echo "      Done!"
echo ""

# Step 2: Install library and target config
echo "[2/4] Installing to CUDA-Q..."
cp libnvqir-einsum.so "${CUDAQ_LIB}/"
echo "      Library: ${CUDAQ_LIB}/libnvqir-einsum.so"
cp "$PROJECT_DIR/targets/einsum.yml" "${CUDAQ_TARGETS}/"
echo "      Config:  ${CUDAQ_TARGETS}/einsum.yml"
echo ""

# Step 3: Verify
echo "[3/4] Verifying installation..."
cd "$PROJECT_DIR"
python3 -c "
import cudaq
cudaq.set_target('einsum')
print('      cudaq.set_target(\"einsum\") - OK!')
"
echo ""

# Step 4: Run basic test
echo "[4/4] Running basic test..."
python3 -c "
import cudaq

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])

cudaq.set_target('einsum')
result = cudaq.sample(bell, shots_count=10)
print('      Bell state circuit executed - OK!')
"
echo ""

echo "============================================================"
echo "  SUCCESS! Einsum simulator is ready."
echo "============================================================"
echo ""
echo "Now you can run examples:"
echo "  python3 examples/01_basic_extraction.py"
echo "  python3 examples/04_qsvm_kernel.py"
echo "  python3 tests/test_capture.py"
echo ""
