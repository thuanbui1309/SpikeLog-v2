"""
Conversion training for SpikeLog-v2: load s1 (LayerNorm) -> convert norms -> optionally fine-tune.

Two modes (set via config["conversion"]["mode"]):
    finetune:     calibrate -> replace LN->tdBN -> fine-tune -> save
    posttraining: calibrate -> replace LN->fixed affine -> save (no training)

Called from run_variant.py when config["conversion"]["enabled"] is true.
"""

import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from spikingjelly.activation_based import functional as sj_functional

from src.data.embedding import generate_event_vectors, load_event_vectors
from src.data.dataset import PairwiseTrainDataset, collate_train
from src.models.factory import create_model
from src.models.tdbn import ThresholdBatchNorm
from src.models.conversion import (
    calibrate_layernorms,
    replace_layernorms_with_tdbn,
    replace_layernorms_with_fixed_affine,
)
from src.utils.logger import TrainingLogger
from src.utils.common import seed_everything, get_device, load_config

log = logging.getLogger(__name__)


def train_convert(config: dict, project_root: str) -> str:
    """Conversion training pipeline.

    Steps:
        1. Load source model (s1) from checkpoint
        2. Calibrate LayerNorm statistics
        3. Replace norms (tdBN or fixed affine)
        4. Fine-tune (finetune mode) or save directly (posttraining mode)
        5. Save best checkpoint
    """
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    variant_cfg = config["variant"]
    convert_cfg = config["conversion"]

    dataset = ds_cfg["name"]
    variant_id = variant_cfg["id"]
    mode = convert_cfg["mode"]  # "finetune" or "posttraining"

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    model_dir = os.path.join(project_root, "results", dataset, variant_id)
    os.makedirs(model_dir, exist_ok=True)

    # -- Step 1: Load source model (s1) ----------------------------------------
    source_variant = convert_cfg["source_variant"]
    source_path = os.path.join(
        project_root, "results", dataset, source_variant, "best_model.pth"
    )
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Source model not found: {source_path}\n"
            f"Train {source_variant} on {dataset} first."
        )

    # Build source config to create model with correct architecture
    source_variant_yaml = os.path.join(
        project_root, "configs", "variants", f"{source_variant}.yaml"
    )
    source_config = load_config(
        os.path.join(project_root, "configs", "base.yaml"),
        source_variant_yaml,
        os.path.join(project_root, "configs", "datasets", f"{dataset}.yaml"),
    )

    device = get_device()
    model = create_model(source_config).to(device)
    model.load_state_dict(torch.load(source_path, map_location=device))
    model.eval()

    print(f"[Convert] Source: {source_variant} loaded from {source_path}")
    print(f"[Convert] Mode: {mode}")

    # -- Step 2: Build dataloader for calibration (and fine-tuning) -------------
    vectors_file = generate_event_vectors(config, project_root)
    event_vectors = load_event_vectors(config, project_root)

    train_normal_file = os.path.join(output_dir, "train_normal.pkl")
    train_anomaly_file = os.path.join(output_dir, "train_anomaly.pkl")

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

    # -- Step 3: Calibrate LayerNorm statistics --------------------------------
    n_cal = convert_cfg.get("calibration_batches", 100)
    print(f"\n[Step 1] Calibrating LayerNorm statistics ({n_cal} batches)...")
    pop_stats = calibrate_layernorms(model, loader, device, n_batches=n_cal)

    # -- Step 4: Replace norms -------------------------------------------------
    best_path = os.path.join(model_dir, "best_model.pth")

    if mode == "posttraining":
        print(f"\n[Step 2] Replacing LayerNorm -> FixedAffineNorm (no training)...")
        model = replace_layernorms_with_fixed_affine(model, pop_stats)
        model = model.to(device)
        torch.save(model.state_dict(), best_path)
        torch.save(pop_stats, os.path.join(model_dir, "calibration_stats.pth"))
        print(f"\n[Done] Post-training conversion saved to: {best_path}")
        return best_path

    elif mode == "finetune":
        print(f"\n[Step 2] Replacing LayerNorm -> tdBN (warm-started)...")
        model = replace_layernorms_with_tdbn(model, pop_stats)
        model = model.to(device)

        # -- Step 5: Fine-tune -------------------------------------------------
        seed_everything(train_cfg.get("seed", 42))
        lr = train_cfg.get("lr", 1e-5)
        weight_decay = train_cfg.get("weight_decay", 0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        grad_clip = train_cfg.get("grad_clip", 1.0)

        max_epoch = train_cfg.get("max_epoch", 20)
        patience = train_cfg.get("patience", 10)
        bn_freeze_epoch = train_cfg.get("bn_freeze_epoch", 0)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )

        train_logger = TrainingLogger(model_dir, variant_id)

        best_loss = float("inf")
        epochs_no_improve = 0
        nan_count = 0

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Step 3] Fine-tuning ({max_epoch} epochs, lr={lr})")
        print(f"  params: {n_params:,}")
        print(f"  train pairs per epoch: {len(train_ds)}")

        for epoch in range(1, max_epoch + 1):
            if bn_freeze_epoch > 0 and epoch == bn_freeze_epoch + 1:
                _freeze_bn(model)
                print(f"  [BN] Froze BatchNorm at epoch {epoch}")

            loss_val = _finetune_epoch(
                model, loader, optimizer, criterion, grad_clip, device, epoch, max_epoch
            )

            if torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val)):
                nan_count += 1
                print(f"  [!] NaN loss at epoch {epoch} (count={nan_count})")
                if os.path.exists(best_path):
                    model.load_state_dict(torch.load(best_path, map_location=device))
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.5
                    print(f"  [!] Reloaded best, LR -> {optimizer.param_groups[0]['lr']:.2e}")
                if nan_count >= 3:
                    print("  [!] 3 NaN events -- stopping early")
                    break
                continue

            nan_count = 0

            marker = ""
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(model.state_dict(), best_path)
                epochs_no_improve = 0
                marker = " * saved"
            else:
                epochs_no_improve += 1

            cur_lr = optimizer.param_groups[0]["lr"]
            train_logger.log_epoch(epoch, {
                "train_loss": loss_val,
                "best_loss": best_loss,
                "lr": cur_lr,
            })
            print(f"  Epoch {epoch:3d}/{max_epoch} | loss={loss_val:.4f} | "
                  f"best={best_loss:.4f} | lr={cur_lr:.1e}{marker}")

            scheduler.step(loss_val)

            if epochs_no_improve >= patience:
                print(f"  [Early stop] no improvement for {patience} epochs")
                break

        # Switch tdBN to running-stats mode for inference
        for m in model.modules():
            if isinstance(m, ThresholdBatchNorm):
                m.use_batch_stats = False

        torch.save(pop_stats, os.path.join(model_dir, "calibration_stats.pth"))
        train_logger.save_charts()

        print(f"\n[Done] Fine-tune conversion complete. Best loss={best_loss:.4f}")
        print(f"    Model saved to: {best_path}")
        return best_path

    else:
        raise ValueError(f"Unknown conversion mode: {mode!r} (expected 'finetune' or 'posttraining')")


def _finetune_epoch(model, loader, optimizer, criterion, grad_clip, device, epoch, max_epoch):
    """One epoch of fine-tuning the converted model."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}/{max_epoch}", leave=False,
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
        return float("nan")
    return total_loss / n_batches


def _freeze_bn(model: nn.Module):
    """Switch all BN-like layers to eval mode (frozen running stats)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, ThresholdBatchNorm)):
            m.eval()
