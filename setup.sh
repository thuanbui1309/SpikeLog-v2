#!/bin/bash
# Setup script for SpikeLog-v2
# Usage:
#   bash setup.sh                  # all datasets
#   bash setup.sh bgl              # BGL only
#   bash setup.sh hdfs bgl         # HDFS and BGL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine Python runner
if command -v uv &> /dev/null; then
    PY="uv run python"
else
    PY="python"
fi

# Determine which datasets to setup
if [ $# -eq 0 ]; then
    DATASETS=("hdfs" "bgl" "thunderbird" "spirit")
else
    DATASETS=("$@")
    DATASETS=("${DATASETS[@],,}")
fi

echo "========================================="
echo "SpikeLog-v2 Setup"
echo "Datasets: ${DATASETS[*]}"
echo "========================================="

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "  Using uv..."
    uv sync
else
    echo "  Using pip..."
    pip install -e .
fi

# Step 2: Download + preprocess datasets
echo ""
echo "[2/3] Preparing datasets..."
for DS in "${DATASETS[@]}"; do
    TRAIN_FILE="data/processed/$DS/train_normal.pkl"
    if [ -f "$TRAIN_FILE" ]; then
        echo "  [✓] $DS already preprocessed"
    else
        DATASET_CFG="configs/datasets/${DS}.yaml"
        if [ ! -f "$DATASET_CFG" ]; then
            echo "  [✗] Config not found: $DATASET_CFG"
            continue
        fi
        echo "  [→] Preparing $DS..."
        $PY -c "
from src.utils.common import load_config
from src.data.download import download_dataset
from src.data.preprocess import prepare_dataset

config = load_config('configs/base.yaml', 'configs/variants/s0_spikelog_baseline.yaml', '$DATASET_CFG')
ds_cfg = config['dataset']
download_dataset(ds_cfg, config['data']['raw_dir'], '.')
prepare_dataset(config, '.')
"
    fi
done

# Step 3: Generate event embeddings (Word2Vec + TF-IDF, shared across variants)
echo ""
echo "[3/3] Generating event embeddings..."
for DS in "${DATASETS[@]}"; do
    VECTORS_FILE="data/processed/$DS/event_vectors.npy"
    if [ -f "$VECTORS_FILE" ]; then
        echo "  [✓] $DS embeddings already generated"
    else
        DATASET_CFG="configs/datasets/${DS}.yaml"
        if [ ! -f "$DATASET_CFG" ]; then
            continue
        fi
        echo "  [→] Generating embeddings for $DS..."
        $PY -c "
from src.utils.common import load_config
from src.data.embedding import generate_event_vectors

config = load_config('configs/base.yaml', 'configs/variants/s0_spikelog_baseline.yaml', '$DATASET_CFG')
generate_event_vectors(config, '.')
"
    fi
done

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  bash run.sh --pending --dataset bgl       # run all variants on BGL"
echo "  bash run.sh s0_spikelog_baseline --dataset bgl  # reproduce SpikeLog baseline"
echo "  bash run.sh --list                        # show status"
