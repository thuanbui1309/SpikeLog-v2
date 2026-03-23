#!/usr/bin/env python
"""
Convert s1 (LayerNorm) model → neuromorphic-compatible via replace + fine-tune.

Usage:
    bash convert.sh --dataset bgl
    bash convert.sh --dataset bgl --finetune-epochs 30 --finetune-lr 1e-5

Pipeline:
    1. Load trained s1 model (SDSA + LayerNorm)
    2. Calibration pass: collect per-channel stats at each LayerNorm
    3. Replace LayerNorm → ThresholdBatchNorm (tdBN)
       - gamma/beta ← LayerNorm gamma/beta
       - running_mean/running_var ← calibration per-channel stats (warm start)
    4. Fine-tune with low LR (model already has good representations)
    5. Evaluate on test set
    6. Save converted model + results

Why this works:
    Training tdBN from scratch diverges because random init + BN noise compounds.
    Fine-tuning from s1 starts with good representations — only needs small adjustments.
    Running stats are warm-started from calibration, so no cold-start problem.
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
from src.models.tdbn import ThresholdBatchNorm
from src.data.embedding import load_event_vectors
from src.data.dataset import PairwiseTrainDataset, PairwiseTestDataset, collate_train, collate_test
from src.utils.common import seed_everything, get_device, load_config


# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate_layernorms(model, dataloader, device, n_batches=100):
    """Collect per-channel population statistics at each LayerNorm.

    Returns:
        dict: name → (channel_mean, channel_var) — both (D,) tensors
    """
    ln_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }
    print(f"  Found {len(ln_modules)} LayerNorm layers to calibrate")

    stats = {name: {"sum": None, "sum_sq": None, "count": 0} for name in ln_modules}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach()
            flat = x.reshape(-1, x.shape[-1])  # (N, D)
            n = flat.shape[0]
            s = stats[name]
            batch_sum = flat.sum(dim=0).cpu()
            batch_sum_sq = (flat ** 2).sum(dim=0).cpu()
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

    model.eval()
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(tqdm(dataloader, desc="  Calibrating", total=n_batches)):
            if i >= n_batches:
                break
            x1, x2 = x1.to(device), x2.to(device)
            sj_functional.reset_net(model.encoder)
            model.encoder(x1)
            sj_functional.reset_net(model.encoder)
            model.encoder(x2)

    for h in hooks:
        h.remove()

    pop_stats = {}
    for name in ln_modules:
        s = stats[name]
        ch_mean = s["sum"] / s["count"]
        ch_var = s["sum_sq"] / s["count"] - ch_mean ** 2
        ch_var = ch_var.clamp(min=1e-8)
        pop_stats[name] = (ch_mean, ch_var)
        print(f"    {name}: mean=[{ch_mean.min():.4f}, {ch_mean.max():.4f}], "
              f"var=[{ch_var.min():.4f}, {ch_var.max():.4f}]")

    return pop_stats


# ── Replace LayerNorm → tdBN ─────────────────────────────────────────────────

def replace_layernorms_with_tdbn(model, pop_stats):
    """Replace each LayerNorm with ThresholdBatchNorm, warm-started from calibration.

    Transfers:
        - gamma, beta ← LayerNorm learnable params
        - running_mean, running_var ← calibration per-channel stats
    """
    for name, (ch_mean, ch_var) in pop_stats.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        ln = getattr(parent, parts[-1])
        assert isinstance(ln, nn.LayerNorm), f"{name} is not LayerNorm"

        D = ln.normalized_shape[0]
        tdbn = ThresholdBatchNorm(D, use_batch_stats=True)  # standard BN for fine-tuning

        # Transfer learnable params
        tdbn.weight.data.copy_(ln.weight.data)  # gamma
        tdbn.bias.data.copy_(ln.bias.data)       # beta

        # Warm-start running stats from calibration
        tdbn.running_mean.copy_(ch_mean)
        tdbn.running_var.copy_(ch_var)

        setattr(parent, parts[-1], tdbn)
        print(f"    {name}: LayerNorm → tdBN (warm-started)")

    n_remaining = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    print(f"  Replaced {len(pop_stats)} layers ({n_remaining} LayerNorm remaining)")
    return model


# ── Fine-tune ────────────────────────────────────────────────────────────────

def finetune(model, dataloader, device, epochs=20, lr=1e-5, grad_clip=1.0):
    """Fine-tune converted model with low LR to adapt to tdBN normalization."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"  Epoch {epoch:3d}/{epochs}", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for x1, x2, y in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            score = model(x1, x2).squeeze(-1)
            loss = criterion(score, y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        pbar.close()

        if n_batches == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | NaN — skipping")
            continue

        avg_loss = total_loss / n_batches
        cur_lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ✓ best"
        else:
            epochs_no_improve += 1

        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | best={best_loss:.4f} | lr={cur_lr:.1e}{marker}")

        scheduler.step(avg_loss)

        if epochs_no_improve >= patience:
            print(f"  [Early stop] no improvement for {patience} epochs")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    print(f"  Fine-tune complete. Best loss={best_loss:.4f}")
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
        for x_test, x_anom, x_norm, labels in tqdm(loader, desc="  Scoring"):
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
    B, n_cmp, ref_T, D = x_refs.shape
    T = x_test.size(1)
    x_test_exp = x_test.unsqueeze(1).expand(B, n_cmp, T, D)
    x_test_flat = x_test_exp.reshape(B * n_cmp, T, D)
    x_refs_flat = x_refs.reshape(B * n_cmp, ref_T, D)
    scores = model(x_test_flat, x_refs_flat).squeeze(-1)
    return scores.reshape(B, n_cmp).mean(dim=1)


def _threshold_search(scores, labels, n_thresholds=200, beta=1.0):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_score = best_f1 = best_prec = best_rec = 0.0
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
    parser = argparse.ArgumentParser(description="Convert s1 → neuromorphic via replace + fine-tune")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-cal", type=int, default=100, help="Calibration batches")
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--source-variant", default="s1_spikelog_sdsa")
    parser.add_argument("--output-variant", default=None)
    args = parser.parse_args()

    output_variant = args.output_variant or f"{args.source_variant}_neuromorphic"

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
        sys.exit(1)

    model = create_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Converting: {args.source_variant} → {output_variant}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {n_params:,} params")
    print(f"Fine-tune:  {args.finetune_epochs} epochs, lr={args.finetune_lr}")
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
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_train, num_workers=2, pin_memory=True,
        drop_last=True,
    )

    pop_stats = calibrate_layernorms(model, train_loader, device, n_batches=args.n_cal)

    # ── Step 3: Replace LayerNorm → tdBN ─────────────────────────────────
    print(f"\n[Step 2] Replacing LayerNorm → tdBN (warm-started)...")
    model = replace_layernorms_with_tdbn(model, pop_stats)
    model = model.to(device)

    # ── Step 4: Fine-tune ────────────────────────────────────────────────
    print(f"\n[Step 3] Fine-tuning ({args.finetune_epochs} epochs, lr={args.finetune_lr})...")
    model = finetune(
        model, train_loader, device,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        grad_clip=1.0,
    )

    # Switch tdBN to running-stats mode for inference (no batch dependency)
    for m in model.modules():
        if isinstance(m, ThresholdBatchNorm):
            m.use_batch_stats = False

    # ── Step 5: Evaluate ─────────────────────────────────────────────────
    print(f"\n[Step 4] Evaluating converted model...")
    results, all_scores, all_labels = evaluate(model, config, PROJECT_ROOT, device)

    # ── Step 6: Save ─────────────────────────────────────────────────────
    result_dir = os.path.join(PROJECT_ROOT, "results", args.dataset, output_variant)
    os.makedirs(result_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
    torch.save(pop_stats, os.path.join(result_dir, "calibration_stats.pth"))

    with open(os.path.join(result_dir, "detection_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(result_dir, "scores.npz"),
             scores=all_scores, labels=all_labels)

    print(f"\n{'='*60}")
    print(f"[Done] {output_variant} on {args.dataset}")
    print(f"  F1={results['f1']:.4f}  P={results['precision']:.4f}  R={results['recall']:.4f}")
    print(f"  Saved to: {result_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
