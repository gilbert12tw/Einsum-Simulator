#!/bin/bash
# Build, install, and test einsum simulator (run inside Docker container or locally)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  Einsum NVQIR - Build, Install, and Test"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
echo "[1/3] Building and Installing via pip..."
python3 -m pip install .
echo ""

echo "[2/3] Installing CUDA-Q target..."
python3 -c "import cudaq_einsum; cudaq_einsum.install_cudaq_target()"
echo ""

echo "[3/3] Running tests..."
python3 tests/test_einsum.py
python3 tests/test_sidecar.py

echo ""
echo "============================================================"
echo "  SUCCESS! Einsum simulator is ready and tested."
echo "============================================================"
