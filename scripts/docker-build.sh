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

# Run build in container (fast - tools already installed)
docker run --rm \
    -v "$PROJECT_DIR:$CONTAINER_PROJECT_DIR" \
    -w "$CONTAINER_PROJECT_DIR" \
    "$IMAGE_NAME" \
    -c '
        set -e
        echo ">>> Creating build directory..."
        mkdir -p build
        cd build

        echo ">>> Running CMake..."
        cmake .. -DCUDAQ_ROOT=/opt/nvidia/cudaq

        echo ">>> Building..."
        make -j$(nproc) VERBOSE=1

        echo ">>> Build complete!"
        ls -la libnvqir-einsum.so
    '

echo "=== Build finished ==="
echo "Output: $PROJECT_DIR/build/libnvqir-einsum.so"
ls -la "$PROJECT_DIR/build/libnvqir-einsum.so"
