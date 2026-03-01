#!/bin/bash
# Build einsum_nvqir using Docker (fast - no package installation each time)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="einsum-nvqir-dev"
CONTAINER_PROJECT_DIR="/workspace/einsum_nvqir"

echo "=== Building einsum_nvqir in Docker ==="
echo "Project directory: $PROJECT_DIR"

# Check if dev image exists, if not build it
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo ">>> Development image not found. Building it first..."
    echo "    (This only happens once)"
    echo ""
    "$SCRIPT_DIR/docker-setup.sh"
    echo ""
fi

# Docker build and test are now unified.
echo "Note: In the new pip-based workflow, building and testing in Docker are combined."
echo "Delegating to docker-test.sh..."
exec "$SCRIPT_DIR/docker-test.sh"
