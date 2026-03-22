"""
Visualize SDSA attention maps for RQ4 (Interpretability).

Usage:
    python visualize_attention.py --dataset bgl --variant s3_spikelog_pruned
    python visualize_attention.py --dataset bgl --variant s2_spikelog_bspn --n_samples 6

Outputs:
    results/{dataset}/{variant}/attention_normal.png
    results/{dataset}/{variant}/attention_anomaly.png
    results/{dataset}/{variant}/attention_comparison.png
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils.common import load_config, get_device, seed_everything
from src.data.embedding import load_event_vectors
from src.models.factory import create_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Visualize SDSA attention maps")
    parser.add_argument("--dataset", required=True, choices=["bgl", "hdfs", "thunderbird"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples per class")
    parser.add_argument("--layer", type=int, default=-1, help="SDSA layer index (-1 = last)")
    args = parser.parse_args()

    seed_everything(42)

    # Load config
    config = load_config(
        os.path.join(PROJECT_ROOT, "configs", "base.yaml"),
        os.path.join(PROJECT_ROOT, "configs", "variants", f"{args.variant}.yaml"),
        os.path.join(PROJECT_ROOT, "configs", "datasets", f"{args.dataset}.yaml"),
    )

    device = get_device()
    dataset = args.dataset
    variant_id = args.variant

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, config["data"]["output_dir"], dataset)
    model_dir = os.path.join(PROJECT_ROOT, "results", dataset, variant_id)
    best_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(best_path):
        print(f"Error: no trained model at {best_path}")
        sys.exit(1)

    # Load model
    model = create_model(config).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Load data
    event_vectors = load_event_vectors(config, PROJECT_ROOT)
    max_seq_len = config["data"].get("max_seq_len", 100)

    with open(os.path.join(data_dir, "test.pkl"), "rb") as f:
        test_data = pickle.load(f)

    # Separate normal and anomaly
    normal_samples = [(seq, label) for seq, label in test_data if label == 0]
    anomaly_samples = [(seq, label) for seq, label in test_data if label == 1]

    print(f"Test: {len(normal_samples)} normal, {len(anomaly_samples)} anomaly")

    # Sample
    n = args.n_samples
    rng = np.random.RandomState(42)
    normal_idx = rng.choice(len(normal_samples), size=min(n, len(normal_samples)), replace=False)
    anomaly_idx = rng.choice(len(anomaly_samples), size=min(n, len(anomaly_samples)), replace=False)

    normal_seqs = [normal_samples[i][0] for i in normal_idx]
    anomaly_seqs = [anomaly_samples[i][0] for i in anomaly_idx]

    # Extract attention
    layer_idx = args.layer
    normal_attns = [extract_attention(model, seq, event_vectors, max_seq_len, device, layer_idx)
                    for seq in normal_seqs]
    anomaly_attns = [extract_attention(model, seq, event_vectors, max_seq_len, device, layer_idx)
                     for seq in anomaly_seqs]

    # Plot
    os.makedirs(model_dir, exist_ok=True)

    plot_attention_grid(normal_attns, normal_seqs, "Normal Sequences",
                        os.path.join(model_dir, "attention_normal.png"))
    plot_attention_grid(anomaly_attns, anomaly_seqs, "Anomaly Sequences",
                        os.path.join(model_dir, "attention_anomaly.png"))
    plot_comparison(normal_attns, anomaly_attns, normal_seqs, anomaly_seqs,
                    os.path.join(model_dir, "attention_comparison.png"))

    print(f"\nSaved to {model_dir}/attention_*.png")


def extract_attention(model, seq, event_vectors, max_seq_len, device, layer_idx=-1):
    """Forward pass a single sequence through the encoder and return attention weights.

    Returns:
        attn: (n_heads, seq_len, seq_len) numpy array
    """
    from spikingjelly.activation_based import functional as sj_functional

    seq = seq[:max_seq_len]
    vecs = event_vectors[seq]  # (seq_len, emb_dim)
    x = torch.from_numpy(vecs.copy()).unsqueeze(0).to(device)  # (1, seq_len, emb_dim)

    with torch.no_grad():
        sj_functional.reset_net(model.encoder)
        _ = model.encoder(x)  # forward pass stores attention in _attn_weights

    # Get attention from specified SDSA block
    blocks = model.encoder.blocks
    block = blocks[layer_idx]
    attn = block.attn._attn_weights  # (1, n_heads, L, L)

    if attn is None:
        raise RuntimeError("No attention weights stored. Check SDSA._attn_weights.")

    return attn[0].cpu().numpy()  # (n_heads, L, L)


def plot_attention_grid(attns, seqs, title, save_path):
    """Plot attention heatmaps for multiple samples (head-averaged)."""
    n = len(attns)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (attn, seq) in enumerate(zip(attns, seqs)):
        # Average over heads
        attn_avg = attn.mean(axis=0)  # (L, L)
        seq_len = len(seq)
        attn_avg = attn_avg[:seq_len, :seq_len]  # crop to actual length

        im = axes[i].imshow(attn_avg, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Sample {i+1} (len={seq_len})", fontsize=10)
        axes[i].set_xlabel("Key position")
        if i == 0:
            axes[i].set_ylabel("Query position")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_comparison(normal_attns, anomaly_attns, normal_seqs, anomaly_seqs, save_path):
    """Side-by-side comparison: normal vs anomaly attention patterns."""
    n = min(len(normal_attns), len(anomaly_attns), 4)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Normal
        attn_n = normal_attns[i].mean(axis=0)
        sl_n = len(normal_seqs[i])
        attn_n = attn_n[:sl_n, :sl_n]
        im = axes[0, i].imshow(attn_n, cmap="viridis", aspect="auto")
        axes[0, i].set_title(f"Normal {i+1} (len={sl_n})", fontsize=10)
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Anomaly
        attn_a = anomaly_attns[i].mean(axis=0)
        sl_a = len(anomaly_seqs[i])
        attn_a = attn_a[:sl_a, :sl_a]
        im = axes[1, i].imshow(attn_a, cmap="viridis", aspect="auto")
        axes[1, i].set_title(f"Anomaly {i+1} (len={sl_a})", fontsize=10)
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Normal\n← Query", fontsize=12)
    axes[1, 0].set_ylabel("Anomaly\n← Query", fontsize=12)

    fig.suptitle("SDSA Attention: Normal vs Anomaly", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


if __name__ == "__main__":
    main()
