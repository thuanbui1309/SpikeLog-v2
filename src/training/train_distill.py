"""
Knowledge distillation training for SpikeLog-v2 BSPN variants.

Teacher: s1 (SDSA + LayerNorm) — already trained, frozen.
Student: s2.1/s3.1 (SDSA + BSPN) — trains with combined loss.

Combined loss:
    L = (1 - α) * L_task + α * L_distill

    L_task    = MSE(score, target)           — standard pairwise regression
    L_distill = MSE(student_repr, teacher_repr)  — representation matching

The teacher guides BSPN student representations toward LayerNorm-quality,
compensating for BSPN's coarser normalization (no centering, quantized scaling).

At inference, the student model is used alone — no teacher needed.
Architecture is unchanged → fully neuromorphic deployable.
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
from src.utils.logger import TrainingLogger
from src.utils.common import seed_everything, get_device, load_config

log = logging.getLogger(__name__)


def train_distill(config: dict, project_root: str):
    """Distillation training pipeline.

    Steps:
        1. Load teacher model (s1) from checkpoint
        2. Create student model (s2.1/s3.1)
        3. Train with combined task + distillation loss
        4. Save best checkpoint
    """
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    variant_cfg = config["variant"]
    distill_cfg = config["distillation"]

    dataset = ds_cfg["name"]
    variant_id = variant_cfg["id"]

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    model_dir = os.path.join(project_root, "results", dataset, variant_id)
    os.makedirs(model_dir, exist_ok=True)

    # ─── Step 1: Load teacher ───────────────────────────────────────────
    teacher_variant = distill_cfg["teacher_variant"]
    teacher_path = os.path.join(
        project_root, "results", dataset, teacher_variant, "best_model.pth"
    )
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(
            f"Teacher model not found: {teacher_path}\n"
            f"Train {teacher_variant} on {dataset} first."
        )

    # Build teacher config by loading teacher variant yaml
    teacher_variant_yaml = os.path.join(
        project_root, "configs", "variants", f"{teacher_variant}.yaml"
    )
    teacher_config = load_config(
        os.path.join(project_root, "configs", "base.yaml"),
        teacher_variant_yaml,
        os.path.join(project_root, "configs", "datasets", f"{dataset}.yaml"),
    )

    device = get_device()
    teacher = create_model(teacher_config).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[Distill] Teacher: {teacher_variant} loaded from {teacher_path}")

    # ─── Step 2: Event vectors + dataset ────────────────────────────────
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

    # ─── Step 3: Student model + optimizer ──────────────────────────────
    seed_everything(train_cfg.get("seed", 42))

    student = create_model(config).to(device)
    lr = train_cfg.get("lr", 5e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    task_criterion = nn.MSELoss()
    distill_criterion = nn.MSELoss()
    grad_clip = train_cfg.get("grad_clip", 5.0)

    alpha = distill_cfg.get("alpha", 0.5)
    max_epoch = train_cfg.get("max_epoch", 50)
    patience = train_cfg.get("patience", 15)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    train_logger = TrainingLogger(model_dir, variant_id)
    best_path = os.path.join(model_dir, "best_model.pth")

    best_loss = float("inf")
    epochs_no_improve = 0
    nan_count = 0

    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"\n[Train] {variant_id} on {dataset} (distill from {teacher_variant})")
    print(f"  student params: {student_params:,} | teacher params: {teacher_params:,}")
    print(f"  alpha={alpha} (distill weight)")
    print(f"  train pairs per epoch: {len(train_ds)}")

    n_batches_total = len(loader)

    for epoch in range(1, max_epoch + 1):
        loss_val, task_loss_val, distill_loss_val = _train_epoch_distill(
            student, teacher, loader, optimizer,
            task_criterion, distill_criterion, alpha,
            grad_clip, device, epoch, max_epoch
        )

        # NaN recovery
        if torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val)):
            nan_count += 1
            print(f"  [!] NaN loss at epoch {epoch} (count={nan_count})")
            if os.path.exists(best_path):
                student.load_state_dict(torch.load(best_path, map_location=device))
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                print(f"  [!] Reloaded best, LR → {optimizer.param_groups[0]['lr']:.2e}")
            if nan_count >= 3:
                print("  [!] 3 NaN events — stopping early")
                break
            continue

        nan_count = 0

        marker = ""
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(student.state_dict(), best_path)
            epochs_no_improve = 0
            marker = " ✓ saved"
        else:
            epochs_no_improve += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        train_logger.log_epoch(epoch, {
            "train_loss": loss_val,
            "task_loss": task_loss_val,
            "distill_loss": distill_loss_val,
            "best_loss": best_loss,
            "lr": cur_lr,
        })
        print(f"  Epoch {epoch:3d}/{max_epoch} | loss={loss_val:.4f} "
              f"(task={task_loss_val:.4f} distill={distill_loss_val:.4f}) | "
              f"best={best_loss:.4f} | lr={cur_lr:.1e}{marker}")

        scheduler.step(loss_val)

        if epochs_no_improve >= patience:
            print(f"  [Early stop] no improvement for {patience} epochs")
            break

    print(f"\n[✓] Distillation training complete. Best loss={best_loss:.4f}")
    print(f"    Model saved to: {best_path}")

    train_logger.save_charts()
    return best_path


def _train_epoch_distill(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task_criterion: nn.Module,
    distill_criterion: nn.Module,
    alpha: float,
    grad_clip: float,
    device: torch.device,
    epoch: int = 0,
    max_epoch: int = 0,
) -> tuple[float, float, float]:
    """One epoch of distillation training.

    Returns:
        (total_loss, task_loss, distill_loss) — epoch averages
    """
    student.train()
    total_loss = 0.0
    total_task = 0.0
    total_distill = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}/{max_epoch}", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for x1, x2, y in pbar:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Student forward — get both score and representations
        sj_functional.reset_net(student.encoder)
        s_r1 = student.encoder(x1)
        sj_functional.reset_net(student.encoder)
        s_r2 = student.encoder(x2)
        s_score = student.output_layer(torch.cat([s_r1, s_r2], dim=1)).squeeze(-1)

        # Teacher forward — get representations (no grad)
        with torch.no_grad():
            sj_functional.reset_net(teacher.encoder)
            t_r1 = teacher.encoder(x1)
            sj_functional.reset_net(teacher.encoder)
            t_r2 = teacher.encoder(x2)

        # Task loss: pairwise regression
        task_loss = task_criterion(s_score, y)

        # Distillation loss: match encoder representations
        distill_loss = (distill_criterion(s_r1, t_r1) + distill_criterion(s_r2, t_r2)) / 2

        # Combined loss
        loss = (1 - alpha) * task_loss + alpha * distill_loss

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_task += task_loss.item()
        total_distill += distill_loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / n_batches, total_task / n_batches, total_distill / n_batches
