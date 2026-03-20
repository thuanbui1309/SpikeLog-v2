"""
Pairwise ordered regression training for SpikeLog-v2.

Loss: MSE between predicted pair score and target y.
    y = 0.0  → normal-normal pair (lowest anomalousness)
    y = 4.0  → normal-anomaly pair
    y = 8.0  → anomaly-anomaly pair (highest anomalousness)

During inference, the anomaly score of a test sample is estimated by comparing
it against reference normal and anomaly sequences (see predict.py).

NaN recovery: if loss is NaN, reload best checkpoint and halve LR (same pattern
as SorLog distill.py — important for BSPN variants s2/s3).
"""

import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.embedding import generate_event_vectors, load_event_vectors
from src.data.dataset import PairwiseTrainDataset, collate_train
from src.models.factory import create_model
from src.utils.logger import TrainingLogger
from src.utils.common import seed_everything, get_device

log = logging.getLogger(__name__)


def train(config: dict, project_root: str):
    """Full training pipeline for one (variant, dataset) combination.

    Steps:
        1. Generate event vectors (if needed)
        2. Build pairwise dataset
        3. Train model with ordered regression loss
        4. Save best checkpoint
    """
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    variant_cfg = config["variant"]

    dataset = ds_cfg["name"]
    variant_id = variant_cfg["id"]

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    model_dir = os.path.join(project_root, "results", dataset, variant_id)
    os.makedirs(model_dir, exist_ok=True)

    # ─── Step 1: Event vectors ─────────────────────────────────────────────
    vectors_file = generate_event_vectors(config, project_root)
    event_vectors = load_event_vectors(config, project_root)

    # ─── Step 2: Dataset ───────────────────────────────────────────────────
    train_normal_file = os.path.join(output_dir, "train_normal.pkl")
    train_anomaly_file = os.path.join(output_dir, "train_anomaly.pkl")

    for f in (train_normal_file, train_anomaly_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found. Run preprocessing first.")

    max_seq_len = data_cfg.get("window_size", 100)
    min_seq_len = data_cfg.get("min_seq_len", 1)

    train_ds = PairwiseTrainDataset(
        train_normal_file, train_anomaly_file, event_vectors, max_seq_len, min_seq_len
    )

    batch_size = train_cfg.get("batch_size", 64)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ─── Step 3: Model + optimizer ─────────────────────────────────────────
    seed_everything(train_cfg.get("seed", 42))
    device = get_device()

    model = create_model(config).to(device)
    lr = train_cfg.get("lr", 5e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    grad_clip = train_cfg.get("grad_clip", 5.0)

    max_epoch = train_cfg.get("max_epoch", 50)
    patience = train_cfg.get("patience", 10)

    # ReduceLROnPlateau: cut LR by 0.5 when loss stops improving (patience=3 epochs)
    # Rationale: this architecture converges fast (2-3 epochs at full LR), then diverges.
    # Plateau reducer cuts LR reactively, allowing continued training at lower LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    train_logger = TrainingLogger(model_dir, variant_id)
    best_path = os.path.join(model_dir, "best_model.pth")

    best_loss = float("inf")
    epochs_no_improve = 0
    nan_count = 0

    print(f"\n[Train] {variant_id} on {dataset}")
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  train pairs per epoch: {len(train_ds)}")

    n_batches_total = len(loader)

    for epoch in range(1, max_epoch + 1):
        loss_val = _train_epoch(model, loader, optimizer, criterion, grad_clip, device,
                                epoch, max_epoch, n_batches_total)

        # NaN recovery
        if torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val)):
            nan_count += 1
            print(f"  [!] NaN loss at epoch {epoch} (count={nan_count})")
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                print(f"  [!] Reloaded best checkpoint, LR → {optimizer.param_groups[0]['lr']:.2e}")
            if nan_count >= 3:
                print("  [!] 3 NaN events — stopping early")
                break
            continue

        nan_count = 0  # reset on clean epoch

        # Save best
        marker = ""
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
            marker = " ✓ saved"
        else:
            epochs_no_improve += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        train_logger.log_epoch(epoch, {"train_loss": loss_val, "best_loss": best_loss, "lr": cur_lr})
        print(f"  Epoch {epoch:3d}/{max_epoch} | loss={loss_val:.4f} | best={best_loss:.4f} | lr={cur_lr:.1e}{marker}")

        scheduler.step(loss_val)

        if epochs_no_improve >= patience:
            print(f"  [Early stop] no improvement for {patience} epochs")
            break

    print(f"\n[✓] Training complete. Best loss={best_loss:.4f}")
    print(f"    Model saved to: {best_path}")

    train_logger.save_charts()

    return best_path


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    device: torch.device,
    epoch: int = 0,
    max_epoch: int = 0,
    n_batches_total: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}/{max_epoch}", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for x1, x2, y in pbar:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        score = model(x1, x2).squeeze(-1)  # (B,)
        loss = criterion(score, y)

        if torch.isnan(loss) or torch.isinf(loss):
            # Skip batch-level NaN
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        avg_loss = total_loss / n_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    pbar.close()

    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches
