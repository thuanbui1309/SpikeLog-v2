"""
tdBN: Threshold-Dependent Batch Normalization for SNNs.

Reference: Zheng et al., "Going Deeper With Directly-Trained Larger Spiking
Neural Networks" (AAAI 2021). 500+ citations — standard SNN normalization.

Modified for pairwise SNN training:
    Standard tdBN uses batch statistics for normalization during training.
    This causes training divergence in pairwise SNN tasks because:
    1. Batch stats are noisy (mix of normal/anomaly sequences per batch)
    2. LIF threshold amplifies this noise (binary output)
    3. Cross-sample gradient coupling destabilizes training

    Our fix: always normalize with running statistics (even during training).
    Running stats are still updated via EMA from each batch.
    This eliminates batch-dependent gradient noise while maintaining
    calibrated normalization.

    Gradient becomes: dy/dx = gamma / sqrt(running_var + eps)
    — clean, no cross-sample coupling, similar stability to LayerNorm.

    At inference: identical to standard BN (uses running stats) — no change
    to neuromorphic deployment or weight folding.

Neuromorphic deployment (fold into weights):
    BN(x) = gamma/sqrt(var+eps) * x + (beta - gamma*mu/sqrt(var+eps))
           = W_bn * x + b_bn

    Combined with preceding Linear(W):
        y = W_bn * (W @ x) + b_bn = (W_bn * W) @ x + b_bn

    Then fold b_bn into LIF threshold:
        spike when (W_merged @ x) >= threshold - b_bn

    Result: Linear(W_merged, no bias) + LIF(threshold_adjusted) -> pure neuromorphic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdBatchNorm(nn.Module):
    """Threshold-Dependent Batch Normalization with running-stats training.

    Always normalizes using running statistics (no batch dependency).
    Running stats are updated via EMA during training.

    Args:
        num_features: size of the last dimension (d_model)
        alpha: threshold-dependent scaling (default 1.0, learnable)
        momentum: EMA momentum for running stats (default 0.01)
        eps: epsilon for numerical stability
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        momentum: float = 0.01,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable affine parameters (same as BN)
        self.weight = nn.Parameter(torch.ones(num_features))   # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))    # beta
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Running statistics (not learnable, updated via EMA)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) or (B, D)
        Returns:
            normalized x, same shape
        """
        if x.dim() == 3:
            B, T, D = x.shape
            flat = x.reshape(B * T, D)
        elif x.dim() == 2:
            flat = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Update running stats during training (no grad — not part of forward graph)
        if self.training and flat.size(0) > 1:
            with torch.no_grad():
                batch_mean = flat.mean(0)
                batch_var = flat.var(0, unbiased=False)
                n = flat.size(0)
                # EMA update (same formula as nn.BatchNorm1d)
                self.running_mean.lerp_(batch_mean, self.momentum)
                # Bessel correction for running var (match PyTorch BN behavior)
                self.running_var.lerp_(batch_var * n / (n - 1), self.momentum)
                self.num_batches_tracked += 1

        # ALWAYS normalize with running stats (no batch dependency in the graph)
        out = F.batch_norm(
            flat,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,  # always use running stats
            eps=self.eps,
        )

        if x.dim() == 3:
            out = out.reshape(B, T, D)

        return out * self.alpha

    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, "
                f"alpha={self.alpha.item():.3f}, "
                f"momentum={self.momentum}, eps={self.eps}")


def fold_tdbn_into_linear(linear: nn.Linear, tdbn: ThresholdBatchNorm) -> nn.Linear:
    """Fold tdBN into preceding Linear layer for neuromorphic export.

    Transforms: Linear(W, b) -> tdBN -> LIF
    Into:       Linear(W_new, b_new) -> LIF(threshold_adjusted)

    The caller should then fold b_new into the LIF threshold.

    Args:
        linear: preceding Linear layer
        tdbn: ThresholdBatchNorm to fold

    Returns:
        New Linear layer with BN folded in (has bias).
    """
    alpha = tdbn.alpha.data
    gamma = tdbn.weight.data            # (D,)
    beta = tdbn.bias.data               # (D,)
    mu = tdbn.running_mean              # (D,)
    var = tdbn.running_var              # (D,)
    eps = tdbn.eps

    # BN scale and shift
    std_inv = 1.0 / torch.sqrt(var + eps)
    w_bn = gamma * std_inv * alpha       # (D,)  — per-channel scale
    b_bn = (beta - gamma * mu * std_inv) * alpha  # (D,) — per-channel bias

    # Merge with preceding Linear
    # y = w_bn * (W @ x + b_linear) + b_bn
    #   = (w_bn * W) @ x + (w_bn * b_linear + b_bn)
    W = linear.weight.data               # (D_out, D_in)
    W_new = W * w_bn.unsqueeze(1)        # scale each output row

    has_bias = linear.bias is not None
    if has_bias:
        b_new = w_bn * linear.bias.data + b_bn
    else:
        b_new = b_bn

    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
    new_linear.weight.data = W_new
    new_linear.bias.data = b_new
    return new_linear
