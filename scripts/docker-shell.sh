#!/bin/bash
# Start an interactive Docker shell for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="einsum-nvqir-dev"
CONTAINER_PROJECT_DIR="/workspace/einsum_nvqir"

# Check if dev image exists, if not build it
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo ">>> Development image not found. Building it first..."
    "$SCRIPT_DIR/docker-setup.sh"
    echo ""
fi

echo "=== Starting interactive Docker shell ==="
echo "Project mounted at: ${CONTAINER_PROJECT_DIR}"
echo "CUDA-Q root: /opt/nvidia/cudaq"
echo ""

docker run -it --rm \
    -v "${PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    -w "${CONTAINER_PROJECT_DIR}" \
    ${IMAGE_NAME}