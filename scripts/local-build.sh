#!/bin/bash
# local-build.sh — Build and install the einsum NVQIR simulator locally
#
# Delegates fully to the standard pip install process.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  Einsum NVQIR — Local Build & Install"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

echo "Building and installing via pip..."
python3 -m pip install .

echo ""
echo "Installing CUDA-Q target into environment..."
python3 -c "import cudaq_einsum; cudaq_einsum.install_cudaq_target()"

echo ""
echo "============================================================"
echo "  SUCCESS — Einsum simulator is ready!"
echo "============================================================"
echo ""
echo "You can now run tests with:"
echo "  python3 tests/test_einsum.py"
echo "  python3 tests/test_sidecar.py"
echo ""
