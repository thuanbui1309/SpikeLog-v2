#!/bin/bash
# Convert s1 (LayerNorm) model → neuromorphic-compatible via calibration
# Usage:
#   bash convert.sh --dataset bgl
#   bash convert.sh --dataset bgl --n-cal 200
#   bash convert.sh --dataset bgl --source-variant s1_spikelog_sdsa

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &> /dev/null; then
    uv run python scripts/convert_neuromorphic.py "$@"
else
    python scripts/convert_neuromorphic.py "$@"
fi
