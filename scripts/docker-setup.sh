#!/bin/bash
# Build the development Docker image (only needs to run once)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="einsum-nvqir-dev"

echo "=== Building Development Docker Image ==="
echo "This only needs to be done once (or when Dockerfile.dev changes)"
echo ""

cd "$PROJECT_DIR"

docker build -t "$IMAGE_NAME" -f Dockerfile.dev .

echo ""
echo "=== Done ==="
echo "Image '$IMAGE_NAME' is ready."
echo "You can now use ./scripts/docker-build.sh for fast builds."
