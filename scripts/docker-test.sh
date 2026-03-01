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

        # Execute unified build and test script inside the container
        bash ${CONTAINER_PROJECT_DIR}/scripts/build-and-test.sh
    "

echo "=== Test completed ==="
