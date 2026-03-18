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
        """Generate and save training charts."""
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

        # Loss curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{self.variant_id} Training", fontsize=14)

        # Train/Valid loss
        ax = axes[0]
        if "train_loss" in self.history[0]:
            ax.plot(epochs, [e["train_loss"] for e in self.history],
                    label="Train", marker=".", markersize=3)
        if "valid_loss" in self.history[0]:
            ax.plot(epochs, [e["valid_loss"] for e in self.history],
                    label="Valid", marker=".", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Component losses (distillation)
        ax = axes[1]
        component_keys = ["emb", "rep", "logit", "mlm", "cls"]
        has_components = any(k in self.history[0] for k in component_keys)
        if has_components:
            for key in component_keys:
                if key in self.history[0]:
                    ax.plot(epochs, [e.get(key, 0) for e in self.history],
                            label=f"L_{key}", marker=".", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Component Losses")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No component losses\n(teacher training)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Component Losses")

        plt.tight_layout()
        chart_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Charts saved: {chart_path}")


def save_detection_chart(log_dir: str, variant_id: str, results: dict):
    """Save detection results chart (confusion matrix + threshold curve)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    best = results.get("best", {})
    if not best:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{variant_id} Detection Results", fontsize=14)

    # Confusion matrix
    ax = axes[0]
    cm = np.array([
        [best.get("TN", 0), best.get("FP", 0)],
        [best.get("FN", 0), best.get("TP", 0)],
    ])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (th={best.get('threshold', 0):.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)

    # Metrics summary
    ax = axes[1]
    metrics_text = (
        f"Accuracy:  {best.get('accuracy', 0):.2f}%\n"
        f"Precision: {best.get('precision', 0):.2f}%\n"
        f"Recall:    {best.get('recall', 0):.2f}%\n"
        f"F1:        {best.get('f1', 0):.2f}%\n"
        f"\nThreshold: {best.get('threshold', 0):.2f}\n"
        f"TP={best.get('TP', 0)}  TN={best.get('TN', 0)}\n"
        f"FP={best.get('FP', 0)}  FN={best.get('FN', 0)}"
    )
    ax.text(0.5, 0.5, metrics_text, ha="center", va="center",
            transform=ax.transAxes, fontsize=14, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    ax.set_title("Detection Metrics")
    ax.axis("off")

    plt.tight_layout()
    chart_path = os.path.join(log_dir, "detection_results.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Detection chart saved: {chart_path}")
