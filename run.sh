#!/bin/bash
# Run SpikeLog-v2 experiments
# Usage:
#   bash run.sh --pending                             # run all pending (all datasets)
#   bash run.sh --pending --dataset bgl               # pending for BGL only
#   bash run.sh s0_spikelog_baseline --dataset bgl    # specific variant on BGL
#   bash run.sh s2_spikelog_bspn --dataset bgl --force --lava  # re-run + energy
#   bash run.sh --list                                # show all status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv &> /dev/null; then
    uv run python scripts/run_variant.py "$@"
else
    python scripts/run_variant.py "$@"
fi
