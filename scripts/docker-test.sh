#!/bin/bash
# Test script for einsum_nvqir simulator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="einsum-nvqir-dev"
CONTAINER_PROJECT_DIR="/workspace/einsum_nvqir"

echo "=== Testing einsum_nvqir Simulator ==="
echo "Project directory: ${PROJECT_DIR}"

# Check if the library exists
if [ ! -f "${PROJECT_DIR}/build/libnvqir-einsum.so" ]; then
    echo "Error: libnvqir-einsum.so not found. Run docker-build.sh first."
    exit 1
fi

# Check if dev image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Error: Development image not found. Run docker-build.sh first."
    exit 1
fi

# Run the test inside Docker
docker run --rm \
    -v "${PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    -w "${CONTAINER_PROJECT_DIR}" \
    ${IMAGE_NAME} \
    -c "
        set -e

        # Copy our library to CUDA-Q's lib directory
        cp ${CONTAINER_PROJECT_DIR}/build/libnvqir-einsum.so /opt/nvidia/cudaq/lib/

        # Create a proper YAML target configuration file
        cat > /opt/nvidia/cudaq/targets/einsum.yml << 'EOF'
# Einsum Builder Target Configuration
name: einsum
description: \"Einsum expression builder simulator for tensor network conversion\"
config:
  nvqir-simulation-backend: einsum
  preprocessor-defines: [\"-D CUDAQ_SIMULATION_SCALAR_FP64\"]
EOF

        echo '>>> Library installed:'
        ls -la /opt/nvidia/cudaq/lib/libnvqir-einsum.so

        echo ''
        echo '>>> Target configuration:'
        cat /opt/nvidia/cudaq/targets/einsum.yml

        echo ''
        echo '>>> Running Python test...'
        python3 ${CONTAINER_PROJECT_DIR}/tests/test_einsum.py
    "

echo "=== Test completed ==="
