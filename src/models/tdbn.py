"""
tdBN: Threshold-Dependent Batch Normalization for SNNs.

Reference: Zheng et al., "Going Deeper With Directly-Trained Larger Spiking
Neural Networks" (AAAI 2021). 500+ citations — standard SNN normalization.

How it works:
    Training:  y = BN(x) * α_th
               where BN = standard batch normalization across (B*T) dimension
               α_th = learnable threshold-dependent scaling factor

    Inference: running mean/var are frozen, BN becomes a fixed affine transform.

Neuromorphic deployment (fold into weights):
    BN(x) = γ/√(σ²+ε) * x + (β - γμ/√(σ²+ε))
           = W_bn * x + b_bn

    Combined with preceding Linear(W):
        y = W_bn * (W @ x) + b_bn = (W_bn * W) @ x + b_bn

    Then fold b_bn into LIF threshold:
        spike when (W_merged @ x) >= threshold - b_bn

    Result: Linear(W_merged, no bias) + LIF(threshold_adjusted) → pure neuromorphic.

Why tdBN works better than BSPN for our architecture:
    - BSPN: only scales, no centering → activations have arbitrary offsets
    - tdBN: centers AND scales → clean activations for LIF neurons
    - Both are neuromorphic-deployable after folding
"""

import torch
import torch.nn as nn


class ThresholdBatchNorm(nn.Module):
    """Threshold-Dependent Batch Normalization.

    Wraps nn.BatchNorm1d to handle (B, T, D) sequence input.
    Normalizes across B*T for each feature d (channel-wise BN).

    Args:
        num_features: size of the last dimension (d_model)
        alpha: threshold-dependent scaling (default 1.0, learnable)
        momentum: BN momentum for running stats (default 0.01, low for SNN stability)
        eps: BN epsilon
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        momentum: float = 0.01,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) or (B, D)
        Returns:
            normalized x, same shape
        """
        if x.dim() == 3:
            B, T, D = x.shape
            # Reshape to (B*T, D) for BatchNorm1d
            out = self.bn(x.reshape(B * T, D))
            out = out.reshape(B, T, D)
        elif x.dim() == 2:
            out = self.bn(x)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        return out * self.alpha

    def extra_repr(self) -> str:
        return f"num_features={self.bn.num_features}, alpha={self.alpha.item():.3f}"


def fold_tdbn_into_linear(linear: nn.Linear, tdbn: ThresholdBatchNorm) -> nn.Linear:
    """Fold tdBN into preceding Linear layer for neuromorphic export.

    Transforms: Linear(W, b) → tdBN → LIF
    Into:       Linear(W_new, b_new) → LIF(threshold_adjusted)

    The caller should then fold b_new into the LIF threshold.

    Args:
        linear: preceding Linear layer
        tdbn: ThresholdBatchNorm to fold

    Returns:
        New Linear layer with BN folded in (has bias).
    """
    bn = tdbn.bn
    alpha = tdbn.alpha.data

    # BN parameters (frozen running stats)
    gamma = bn.weight.data          # (D,)
    beta = bn.bias.data             # (D,)
    mu = bn.running_mean            # (D,)
    var = bn.running_var             # (D,)
    eps = bn.eps

    # BN scale and shift
    std_inv = 1.0 / torch.sqrt(var + eps)
    w_bn = gamma * std_inv * alpha   # (D,)  — per-channel scale
    b_bn = (beta - gamma * mu * std_inv) * alpha  # (D,) — per-channel bias

    # Merge with preceding Linear
    # y = w_bn * (W @ x + b_linear) + b_bn
    #   = (w_bn * W) @ x + (w_bn * b_linear + b_bn)
    W = linear.weight.data           # (D_out, D_in)
    W_new = W * w_bn.unsqueeze(1)    # scale each output row

    has_bias = linear.bias is not None
    if has_bias:
        b_new = w_bn * linear.bias.data + b_bn
    else:
        b_new = b_bn

    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
    new_linear.weight.data = W_new
    new_linear.bias.data = b_new
    return new_linear
