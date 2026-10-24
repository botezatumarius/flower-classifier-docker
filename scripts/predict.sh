#!/usr/bin/env bash
# Terminal inference in an isolated, resource-limited container. Usage: scripts/predict.sh [images_dir] [image_tag]
set -euo pipefail

IMAGES_DIR=$(cd "${1:-gui/examples}" && pwd)
IMAGE=${2:-flower-inference:1.0}
GPU_FLAGS=()
[[ -n "${GPU_ID:-}" ]] && GPU_FLAGS=(--gpus "device=${GPU_ID}")

docker run \
  --rm \
  --name "flower-predict-${USER:-local}" \
  "${GPU_FLAGS[@]}" \
  --cpus 4 \
  --memory 2g \
  --memory-swap 2g \
  --shm-size 256m \
  --pids-limit 256 \
  --env OMP_NUM_THREADS=4 \
  --network none \
  --read-only \
  --tmpfs /tmp \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --mount "type=bind,source=${IMAGES_DIR},target=/images,readonly" \
  "${IMAGE}" \
  python -m flowers.predict /images -k 3
