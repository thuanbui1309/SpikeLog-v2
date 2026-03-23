"""
Training logger and chart generator.

Saves per-epoch metrics to JSON and generates loss/metric plots.
Logs are saved alongside model weights in results/{dataset}/{variant_id}/.
"""

import json
import os
from datetime import datetime


class TrainingLogger:
    """Log training metrics per epoch and save to JSON + generate charts."""

    def __init__(self, log_dir: str, variant_id: str):
        self.log_dir = log_dir
        self.variant_id = variant_id
        self.history = []
        self.start_time = None
        os.makedirs(log_dir, exist_ok=True)

    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for one epoch."""
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)
        # Save incrementally (crash-safe)
        self._save_json()

    def _save_json(self):
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "variant_id": self.variant_id,
                "timestamp": datetime.now().isoformat(),
                "epochs": self.history,
            }, f, indent=2)

    def save_charts(self):
        """Generate and save training loss curve with best-loss annotation."""
        if len(self.history) < 2:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [warn] matplotlib not installed, skipping charts")
            return

        epochs = [e["epoch"] for e in self.history]
        train_losses = [e["train_loss"] for e in self.history if "train_loss" in e]

        if not train_losses:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"{self.variant_id} — Training Loss", fontsize=14, fontweight="bold")

        # Main loss curve
        ax.plot(epochs[:len(train_losses)], train_losses,
                color="#1976D2", linewidth=1.5, marker=".", markersize=4, label="Train Loss")

        # Best loss line
        if "best_loss" in self.history[0]:
            best_losses = [e["best_loss"] for e in self.history]
            ax.plot(epochs, best_losses, color="#4CAF50", linewidth=1, linestyle="--",
                    alpha=0.7, label="Best Loss")

        # Annotate final best
        best_loss = min(train_losses)
        best_epoch = epochs[train_losses.index(best_loss)]
        ax.annotate(f"Best: {best_loss:.4f} (ep {best_epoch})",
                    xy=(best_epoch, best_loss), xytext=(best_epoch + 1, best_loss * 1.1),
                    arrowprops=dict(arrowstyle="->", color="#E53935"),
                    fontsize=10, color="#E53935", fontweight="bold")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training chart saved: {chart_path}")


def save_detection_chart(
    log_dir: str,
    variant_id: str,
    results: dict,
    scores: "np.ndarray | None" = None,
    labels: "np.ndarray | None" = None,
):
    """Save detection results: confusion matrix + score distribution + metrics card.

    Args:
        log_dir: directory to save plots
        variant_id: variant name for title
        results: dict with precision, recall, f1, threshold, TP, TN, FP, FN
        scores: (N,) anomaly scores per test sample (optional, for histogram)
        labels: (N,) ground truth labels 0/1 (optional, for histogram)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import f1_score as _f1
    except ImportError:
        print("  [warn] matplotlib not installed, skipping charts")
        return

    has_scores = scores is not None and labels is not None
    n_cols = 3 if has_scores else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    fig.suptitle(f"{variant_id} — Detection Results", fontsize=14, fontweight="bold")

    # ── 1. Confusion Matrix ─────────────────────────────────────────────
    ax = axes[0]
    cm = np.array([
        [results.get("TN", 0), results.get("FP", 0)],
        [results.get("FN", 0), results.get("TP", 0)],
    ])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (th={results.get('threshold', 0):.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── 2. Score Distribution Histogram ──────────────────────────────────
    if has_scores:
        ax = axes[1]
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        bins = np.linspace(scores.min(), scores.max(), 50)
        ax.hist(normal_scores, bins=bins, alpha=0.6, label=f"Normal ({len(normal_scores):,})",
                color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.hist(anomaly_scores, bins=bins, alpha=0.6, label=f"Anomaly ({len(anomaly_scores):,})",
                color="#F44336", edgecolor="white", linewidth=0.5)
        thresh = results.get("threshold", 0)
        ax.axvline(thresh, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold={thresh:.3f}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        metrics_idx = 2
    else:
        metrics_idx = 1

    # ── 3. Metrics Summary Card ──────────────────────────────────────────
    ax = axes[metrics_idx]
    prec = results.get("precision", 0)
    rec = results.get("recall", 0)
    f1 = results.get("f1", 0)
    spec = results.get("specificity", 0)
    tp = results.get("TP", 0)
    tn = results.get("TN", 0)
    fp = results.get("FP", 0)
    fn = results.get("FN", 0)
    n_test = results.get("n_test", tp + tn + fp + fn)
    accuracy = (tp + tn) / max(n_test, 1)

    metrics_text = (
        f"Precision:   {prec * 100:6.2f}%\n"
        f"Recall:      {rec * 100:6.2f}%\n"
        f"F1 Score:    {f1 * 100:6.2f}%\n"
        f"Specificity: {spec * 100:6.2f}%\n"
        f"Accuracy:    {accuracy * 100:6.2f}%\n"
        f"\n"
        f"Threshold:   {results.get('threshold', 0):.4f}\n"
        f"TP={tp:>6,}   TN={tn:>6,}\n"
        f"FP={fp:>6,}   FN={fn:>6,}\n"
        f"\n"
        f"Total test:  {n_test:,}"
    )
    ax.text(0.5, 0.5, metrics_text, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFFDE7", edgecolor="#FBC02D"))
    ax.set_title("Metrics Summary")
    ax.axis("off")

    plt.tight_layout()
    chart_path = os.path.join(log_dir, "detection_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Detection chart saved: {chart_path}")
