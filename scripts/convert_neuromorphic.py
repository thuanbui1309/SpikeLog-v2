#!/usr/bin/env python
"""
Convert s1 (LayerNorm) model → neuromorphic-compatible via calibration.

Usage:
    python scripts/convert_neuromorphic.py --dataset bgl
    python scripts/convert_neuromorphic.py --dataset bgl --n-cal 100

Pipeline:
    1. Load trained s1 model (SDSA + LayerNorm)
    2. Calibration pass: collect per-LayerNorm input statistics over training data
    3. Replace LayerNorm → FixedAffineNorm (constant scale + bias)
    4. Evaluate converted model on test set (same as predict.py)
    5. Save converted model + results

LayerNorm replacement (per-channel, like BN inference):
    Collect per-feature E[x_i] and Var[x_i] over training data.
    Replace LayerNorm with per-channel fixed affine:
        scale_i = gamma_i / sqrt(Var[x_i] + eps)
        bias_i  = beta_i - gamma_i * E[x_i] / sqrt(Var[x_i] + eps)
        y_i = scale_i * x_i + bias_i

    FixedAffineNorm can be folded into adjacent Linear weights for hardware.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from tqdm import tqdm

from spikingjelly.activation_based import functional as sj_functional

# ── Project imports ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.factory import create_model
from src.data.embedding import load_event_vectors
from src.data.dataset import PairwiseTrainDataset, PairwiseTestDataset, collate_train, collate_test
from src.utils.common import seed_everything, get_device, load_config


# ── FixedAffineNorm ──────────────────────────────────────────────────────────

class FixedAffineNorm(nn.Module):
    """Fixed affine replacement for LayerNorm, using calibrated population stats.

    y = scale * x + bias

    Neuromorphic-compatible: no input-dependent statistics at runtime.
    Can be folded into adjacent Linear weights for hardware deployment.
    """

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)  # (D,)
        self.register_buffer("bias", bias)    # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias

    def extra_repr(self) -> str:
        return f"features={self.scale.shape[0]}"


# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate_layernorms(model, dataloader, device, n_batches=50):
    """Collect per-channel population statistics at each LayerNorm.

    For each LayerNorm, computes per-feature E[x_i] and Var[x_i] over the
    calibration set. This converts LayerNorm → BatchNorm-style fixed stats,
    giving a per-channel affine that's much more accurate than scalar approx.

    Returns:
        dict: name → (channel_mean, channel_var) — both (D,) tensors
    """
    ln_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }
    print(f"  Found {len(ln_modules)} LayerNorm layers to calibrate")

    # Per-channel accumulators: sum and sum_sq for Welford-style mean/var
    stats = {name: {"sum": None, "sum_sq": None, "count": 0} for name in ln_modules}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach()
            # Flatten to (N, D) — collect per-channel stats
            flat = x.reshape(-1, x.shape[-1])  # (N, D)
            n = flat.shape[0]
            s = stats[name]
            batch_sum = flat.sum(dim=0).cpu()        # (D,)
            batch_sum_sq = (flat ** 2).sum(dim=0).cpu()  # (D,)
            if s["sum"] is None:
                s["sum"] = batch_sum
                s["sum_sq"] = batch_sum_sq
            else:
                s["sum"] += batch_sum
                s["sum_sq"] += batch_sum_sq
            s["count"] += n
        return hook_fn

    for name, module in ln_modules.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass over calibration data
    model.eval()
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(tqdm(dataloader, desc="Calibrating", total=n_batches)):
            if i >= n_batches:
                break
            x1, x2 = x1.to(device), x2.to(device)
            sj_functional.reset_net(model.encoder)
            model.encoder(x1)
            sj_functional.reset_net(model.encoder)
            model.encoder(x2)

    for h in hooks:
        h.remove()

    # Compute per-channel population statistics
    pop_stats = {}
    for name in ln_modules:
        s = stats[name]
        channel_mean = s["sum"] / s["count"]                           # (D,)
        channel_var = s["sum_sq"] / s["count"] - channel_mean ** 2     # (D,)
        channel_var = channel_var.clamp(min=1e-8)  # numerical safety
        pop_stats[name] = (channel_mean, channel_var)
        print(f"    {name}: ch_mean=[{channel_mean.min():.4f}, {channel_mean.max():.4f}], "
              f"ch_var=[{channel_var.min():.4f}, {channel_var.max():.4f}]")

    return pop_stats


def replace_layernorms(model, pop_stats):
    """Replace each LayerNorm with FixedAffineNorm using calibrated statistics.

    Returns the model with all LayerNorms replaced (in-place).
    """
    for name, (channel_mean, channel_var) in pop_stats.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        ln = getattr(parent, parts[-1])
        assert isinstance(ln, nn.LayerNorm), f"{name} is not LayerNorm"

        gamma = ln.weight.data.clone().cpu()  # (D,)
        beta = ln.bias.data.clone().cpu()     # (D,)
        eps = ln.eps

        # Per-channel fixed affine: y_i = scale_i * x_i + bias_i
        # Approximates LayerNorm using per-channel population stats (like BN inference)
        inv_std = 1.0 / (channel_var + eps).sqrt()  # (D,)
        scale = gamma * inv_std                       # (D,)
        bias = beta - gamma * channel_mean * inv_std  # (D,)

        fixed = FixedAffineNorm(scale, bias)
        setattr(parent, parts[-1], fixed)

    n_remaining = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    print(f"  Replaced {len(pop_stats)} LayerNorms → FixedAffineNorm ({n_remaining} remaining)")
    return model


# ── Evaluation (same logic as predict.py) ────────────────────────────────────

def evaluate(model, config, project_root, device):
    """Run anomaly detection on test set with the converted model."""
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    dataset = ds_cfg["name"]

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    event_vectors = load_event_vectors(config, project_root)

    max_seq_len = data_cfg.get("window_size", 100)
    n_cmp = data_cfg.get("n_comparisons", 30)

    test_ds = PairwiseTestDataset(
        os.path.join(output_dir, "test.pkl"),
        os.path.join(output_dir, "train_normal.pkl"),
        os.path.join(output_dir, "train_anomaly.pkl"),
        event_vectors, n_cmp, max_seq_len,
    )

    batch_size = config["training"].get("batch_size", 64)
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_test, num_workers=2, pin_memory=True,
    )

    print(f"\n[Evaluate] Converted model on {dataset} | {len(test_ds)} test samples")

    all_scores = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x_test, x_anom, x_norm, labels in tqdm(loader, desc="Scoring"):
            x_test = x_test.to(device)
            x_anom = x_anom.to(device)
            x_norm = x_norm.to(device)

            scores_anom = _batch_compare(model, x_test, x_anom)
            scores_norm = _batch_compare(model, x_test, x_norm)
            anomaly_score = (scores_anom + scores_norm) / 2

            all_scores.extend(anomaly_score.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Threshold search
    best_f1, best_prec, best_rec, best_thresh = _threshold_search(all_scores, all_labels)

    best_preds = (all_scores >= best_thresh).astype(int)
    tp = int(((best_preds == 1) & (all_labels == 1)).sum())
    tn = int(((best_preds == 0) & (all_labels == 0)).sum())
    fp = int(((best_preds == 1) & (all_labels == 0)).sum())
    fn = int(((best_preds == 0) & (all_labels == 1)).sum())

    results = {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "threshold": float(best_thresh),
        "n_test": len(all_labels),
        "n_anomaly": int(all_labels.sum()),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

    print(f"\n  Results:")
    print(f"    Threshold:  {best_thresh:.4f}")
    print(f"    Precision:  {best_prec:.4f}")
    print(f"    Recall:     {best_rec:.4f}")
    print(f"    F1:         {best_f1:.4f}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    return results, all_scores, all_labels


def _batch_compare(model, x_test, x_refs):
    """Mean pairwise score between test samples and reference set."""
    B, n_cmp, ref_T, D = x_refs.shape
    T = x_test.size(1)

    x_test_exp = x_test.unsqueeze(1).expand(B, n_cmp, T, D)
    x_test_flat = x_test_exp.reshape(B * n_cmp, T, D)
    x_refs_flat = x_refs.reshape(B * n_cmp, ref_T, D)

    scores = model(x_test_flat, x_refs_flat).squeeze(-1)
    return scores.reshape(B, n_cmp).mean(dim=1)


def _threshold_search(scores, labels, n_thresholds=200, beta=1.0):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_score = 0.0
    best_f1 = best_prec = best_rec = 0.0
    best_thresh = thresholds[0]

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        score = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_f1 = f1_score(labels, preds, zero_division=0)
            best_prec = precision_score(labels, preds, zero_division=0)
            best_rec = recall_score(labels, preds, zero_division=0)
            best_thresh = thresh

    return best_f1, best_prec, best_rec, best_thresh


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert s1 (LayerNorm) → neuromorphic")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. bgl)")
    parser.add_argument("--n-cal", type=int, default=100, help="Calibration batches")
    parser.add_argument("--source-variant", default="s1_spikelog_sdsa",
                        help="Source variant (trained with LayerNorm)")
    parser.add_argument("--output-variant", default=None,
                        help="Output variant ID (default: {source}_neuromorphic)")
    args = parser.parse_args()

    output_variant = args.output_variant or f"{args.source_variant}_neuromorphic"

    # Load config
    config = load_config(
        os.path.join(PROJECT_ROOT, "configs", "base.yaml"),
        os.path.join(PROJECT_ROOT, "configs", "variants", f"{args.source_variant}.yaml"),
        os.path.join(PROJECT_ROOT, "configs", "datasets", f"{args.dataset}.yaml"),
    )

    device = get_device()
    seed_everything(config.get("seed", 1234))

    # ── Step 1: Load trained s1 model ────────────────────────────────────
    model_path = os.path.join(
        PROJECT_ROOT, "results", args.dataset, args.source_variant, "best_model.pth"
    )
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model at {model_path}")
        print(f"Train {args.source_variant} on {args.dataset} first.")
        sys.exit(1)

    model = create_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Converting: {args.source_variant} → {output_variant}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {n_params:,} params")
    print(f"{'='*60}")

    # ── Step 2: Calibration ──────────────────────────────────────────────
    print(f"\n[Step 1] Calibrating LayerNorm statistics ({args.n_cal} batches)...")

    data_cfg = config["data"]
    ds_cfg = config["dataset"]
    output_dir = os.path.join(PROJECT_ROOT, data_cfg["output_dir"], ds_cfg["name"])
    event_vectors = load_event_vectors(config, PROJECT_ROOT)

    max_seq_len = data_cfg.get("window_size", 100)
    min_seq_len = data_cfg.get("min_seq_len", 1)

    train_ds = PairwiseTrainDataset(
        os.path.join(output_dir, "train_normal.pkl"),
        os.path.join(output_dir, "train_anomaly.pkl"),
        event_vectors, max_seq_len, min_seq_len,
    )
    cal_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_train, num_workers=2, pin_memory=True,
    )

    pop_stats = calibrate_layernorms(model, cal_loader, device, n_batches=args.n_cal)

    # ── Step 3: Replace LayerNorms ───────────────────────────────────────
    print(f"\n[Step 2] Replacing LayerNorm → FixedAffineNorm...")
    model = replace_layernorms(model, pop_stats)
    model = model.to(device)

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print(f"\n[Step 3] Evaluating converted model...")
    results, all_scores, all_labels = evaluate(model, config, PROJECT_ROOT, device)

    # ── Step 5: Save ─────────────────────────────────────────────────────
    result_dir = os.path.join(PROJECT_ROOT, "results", args.dataset, output_variant)
    os.makedirs(result_dir, exist_ok=True)

    # Save converted model
    torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))

    # Save calibration stats
    torch.save(pop_stats, os.path.join(result_dir, "calibration_stats.pth"))

    # Save detection results
    with open(os.path.join(result_dir, "detection_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw scores
    np.savez(os.path.join(result_dir, "scores.npz"),
             scores=all_scores, labels=all_labels)

    print(f"\n{'='*60}")
    print(f"[Done] {output_variant} on {args.dataset}")
    print(f"  F1={results['f1']:.4f}  P={results['precision']:.4f}  R={results['recall']:.4f}")
    print(f"  Saved to: {result_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
