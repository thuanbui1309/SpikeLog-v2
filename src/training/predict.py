"""
Pairwise anomaly prediction for SpikeLog-v2.

Inference strategy (from SpikeLog paper):
    For each test sample x:
        1. Compare against n_comparisons anomalous references → scores_anom
        2. Compare against n_comparisons normal references → scores_norm
        3. Anomaly score = mean(scores_anom) - mean(scores_norm)
           (higher = more likely anomalous)
    4. Threshold search: find t that maximizes F1 on test set
       (in practice: try 100 linearly spaced thresholds on [min, max] of scores)

Reference: SpikeLog (Qi et al., IEEE TKDE 2024), predict_pairwised()
"""

import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from tqdm import tqdm

from src.data.embedding import load_event_vectors
from src.data.dataset import PairwiseTestDataset, collate_test
from src.models.factory import create_model
from src.utils.common import get_device

log = logging.getLogger(__name__)


def predict(config: dict, project_root: str):
    """Run anomaly detection on test set and compute metrics.

    Loads the best checkpoint, computes pairwise anomaly scores, searches
    for the best threshold, and saves results to registry + JSON.

    Returns:
        dict with keys: precision, recall, f1, threshold
    """
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    variant_cfg = config["variant"]

    dataset = ds_cfg["name"]
    variant_id = variant_cfg["id"]

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    model_dir = os.path.join(project_root, "results", dataset, variant_id)
    best_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No trained model at {best_path}. Run training first.")

    device = get_device()
    event_vectors = load_event_vectors(config, project_root)

    max_seq_len = data_cfg.get("window_size", 100)
    n_cmp = data_cfg.get("n_comparisons", 30)

    test_file = os.path.join(output_dir, "test.pkl")
    train_normal_file = os.path.join(output_dir, "train_normal.pkl")
    train_anomaly_file = os.path.join(output_dir, "train_anomaly.pkl")

    test_ds = PairwiseTestDataset(
        test_file, train_normal_file, train_anomaly_file,
        event_vectors, n_cmp, max_seq_len
    )

    batch_size = config["training"].get("batch_size", 64)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test,
        num_workers=2,
        pin_memory=True,
    )

    # Load model
    model = create_model(config).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    print(f"\n[Predict] {variant_id} on {dataset} | {len(test_ds)} test samples")

    # ─── Score all test samples ────────────────────────────────────────────
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x_test, x_anom, x_norm, labels in tqdm(loader, desc="Scoring"):
            x_test = x_test.to(device)          # (B, T, D)
            x_anom = x_anom.to(device)          # (B, n_cmp, ref_T, D)
            x_norm = x_norm.to(device)          # (B, n_cmp, ref_T, D)

            B = x_test.size(0)
            scores_anom = _batch_compare(model, x_test, x_anom, device)  # (B,)
            scores_norm = _batch_compare(model, x_test, x_norm, device)  # (B,)

            # Anomaly score: average of both pairwise scores (matches original SpikeLog)
            # Higher score = model thinks test sample participates in more anomalous pairs
            anomaly_score = (scores_anom + scores_norm) / 2  # (B,)
            all_scores.extend(anomaly_score.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ─── Threshold search ─────────────────────────────────────────────────
    # beta=1: standard F1. beta=2: recall-weighted (for anomaly detection).
    threshold_beta = config.get("detection", {}).get("threshold_beta", 1.0)
    best_f1, best_prec, best_rec, best_thresh = _threshold_search(
        all_scores, all_labels, beta=threshold_beta
    )

    # Confusion matrix at best threshold
    best_preds = (all_scores >= best_thresh).astype(int)
    tp = int(((best_preds == 1) & (all_labels == 1)).sum())
    tn = int(((best_preds == 0) & (all_labels == 0)).sum())
    fp = int(((best_preds == 1) & (all_labels == 0)).sum())
    fn = int(((best_preds == 0) & (all_labels == 1)).sum())

    specificity = tn / max(tn + fp, 1)

    print(f"\n  Results:")
    print(f"    Threshold:   {best_thresh:.4f}")
    print(f"    Precision:   {best_prec:.4f}")
    print(f"    Recall:      {best_rec:.4f}")
    print(f"    F1:          {best_f1:.4f}")
    print(f"    Specificity: {specificity:.4f}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # Save results
    results = {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "specificity": float(specificity),
        "threshold": float(best_thresh),
        "n_test": len(all_labels),
        "n_anomaly": int(all_labels.sum()),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }
    with open(os.path.join(model_dir, "detection_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw scores for analysis
    np.savez(os.path.join(model_dir, "scores.npz"),
             scores=all_scores, labels=all_labels)

    # Generate charts
    from src.utils.logger import save_detection_chart
    save_detection_chart(model_dir, variant_id, results, all_scores, all_labels)

    return results


def _batch_compare(
    model: nn.Module,
    x_test: torch.Tensor,
    x_refs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute mean pairwise score between each test sample and its reference set.

    Args:
        x_test : (B, T, D)
        x_refs : (B, n_cmp, ref_T, D)
    Returns:
        mean_score: (B,) — average score over n_cmp comparisons
    """
    B, n_cmp, ref_T, D = x_refs.shape
    T = x_test.size(1)

    # Expand x_test for all n_cmp comparisons: (B*n_cmp, T, D)
    x_test_expanded = x_test.unsqueeze(1).expand(B, n_cmp, T, D)
    x_test_flat = x_test_expanded.reshape(B * n_cmp, T, D)
    x_refs_flat = x_refs.reshape(B * n_cmp, ref_T, D)

    scores = model(x_test_flat, x_refs_flat).squeeze(-1)  # (B*n_cmp,)
    scores = scores.reshape(B, n_cmp)
    return scores.mean(dim=1)  # (B,)


def _threshold_search(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
    beta: float = 1.0,
) -> tuple[float, float, float, float]:
    """Grid search over thresholds to maximize F-beta score.

    Args:
        beta: F-beta parameter. beta=1 → standard F1 (balanced).
              beta=2 → recall-weighted (penalizes missed anomalies more).

    Returns:
        (best_f1, best_precision, best_recall, best_threshold)
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_score = 0.0
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
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
