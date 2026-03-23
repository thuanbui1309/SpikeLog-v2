"""
Shared utilities for LayerNorm -> neuromorphic normalization conversion.

Used by:
    - src/training/train_convert.py  (pipeline-compatible conversion training)
    - scripts/convert_neuromorphic.py (standalone CLI)

Two conversion modes:
    1. finetune:     calibrate LN stats -> replace LN->tdBN -> fine-tune
    2. posttraining: calibrate LN stats -> replace LN->fixed affine -> save (no training)
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from spikingjelly.activation_based import functional as sj_functional

from src.models.tdbn import ThresholdBatchNorm


# -- Calibration ---------------------------------------------------------------

def calibrate_layernorms(model, dataloader, device, n_batches=100):
    """Collect per-channel population statistics at each LayerNorm.

    Runs a forward pass over n_batches of training data and records
    per-channel mean and variance at each LayerNorm input.

    Returns:
        dict: name -> (channel_mean, channel_var) -- both (D,) tensors
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


# -- Replace LayerNorm -> tdBN ------------------------------------------------

def replace_layernorms_with_tdbn(model, pop_stats):
    """Replace each LayerNorm with ThresholdBatchNorm, warm-started from calibration.

    Transfers:
        - gamma, beta <- LayerNorm learnable params
        - running_mean, running_var <- calibration per-channel stats
    """
    for name, (ch_mean, ch_var) in pop_stats.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        ln = getattr(parent, parts[-1])
        assert isinstance(ln, nn.LayerNorm), f"{name} is not LayerNorm"

        D = ln.normalized_shape[0]
        tdbn = ThresholdBatchNorm(D, use_batch_stats=True)

        tdbn.weight.data.copy_(ln.weight.data)
        tdbn.bias.data.copy_(ln.bias.data)
        tdbn.running_mean.copy_(ch_mean)
        tdbn.running_var.copy_(ch_var)

        setattr(parent, parts[-1], tdbn)
        print(f"    {name}: LayerNorm -> tdBN (warm-started)")

    n_remaining = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    print(f"  Replaced {len(pop_stats)} layers ({n_remaining} LayerNorm remaining)")
    return model


# -- FixedAffineNorm (post-training mode) -------------------------------------

class FixedAffineNorm(nn.Module):
    """Fixed affine normalization using pre-computed population statistics.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    No learnable parameters updated during forward -- purely static.
    Used for post-training conversion (no fine-tuning).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


def replace_layernorms_with_fixed_affine(model, pop_stats):
    """Replace each LayerNorm with FixedAffineNorm using calibration stats.

    No trainable parameters -- purely static normalization.
    """
    for name, (ch_mean, ch_var) in pop_stats.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        ln = getattr(parent, parts[-1])
        assert isinstance(ln, nn.LayerNorm), f"{name} is not LayerNorm"

        D = ln.normalized_shape[0]
        fixed = FixedAffineNorm(D)

        fixed.weight.copy_(ln.weight.data)
        fixed.bias.copy_(ln.bias.data)
        fixed.running_mean.copy_(ch_mean)
        fixed.running_var.copy_(ch_var)

        setattr(parent, parts[-1], fixed)
        print(f"    {name}: LayerNorm -> FixedAffineNorm")

    n_remaining = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    print(f"  Replaced {len(pop_stats)} layers ({n_remaining} LayerNorm remaining)")
    return model
