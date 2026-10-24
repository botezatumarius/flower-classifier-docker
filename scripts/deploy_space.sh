#!/usr/bin/env bash
# Publishes the inference service as a Docker Space. Usage: scripts/deploy_space.sh <user>/<space> (after `hf auth login`)
set -euo pipefail

SPACE=${1:?usage: scripts/deploy_space.sh <user>/<space>}

hf repos create "$SPACE" --type space --sdk docker --exist-ok
hf upload "$SPACE" inference . --type space --exclude "**/__pycache__/*" --commit-message "Deploy flower classifier API"
echo "API: https://${SPACE/\//-}.hf.space/docs"
