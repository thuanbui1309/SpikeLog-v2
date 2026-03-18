"""
BSPN: Bit-Shifting PowerNorm — neuromorphic-compatible normalization.

Reference: Sorbet (ICML 2025) — replaces LayerNorm with integer-only operations.

How it works:
    1. Compute L1 norm of input: ||x||_1 / d
    2. Round to nearest power-of-2: 2^round(log2(norm))
    3. Normalize by bit-shifting: x >> shift_amount
    4. Apply learnable scale (gamma)

Why neuromorphic-compatible:
    - No division (replaced by bit shift)
    - No square root (L1 instead of L2)
    - No running statistics (unlike BatchNorm)
    - All operations: addition, comparison, bit shift → works on Loihi 2

Training mode:
    - Uses straight-through estimator (STE) for the rounding operation
    - Gamma is learnable, applied as multiplication during training
    - During Lava export, gamma is folded into preceding layer weights

Comparison with LayerNorm:
    - LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    - BSPN:     y = (x >> shift) * gamma
    - BSPN has no beta (bias-free for neuromorphic), no mean subtraction
"""

import torch
import torch.nn as nn
import math


class BitShiftPowerNorm(nn.Module):
    """Bit-Shifting PowerNorm for neuromorphic-compatible normalization.

    Args:
        normalized_shape: size of the last dimension to normalize
        eps: small constant to avoid log2(0)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Learnable scale parameter (no bias for neuromorphic compatibility)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        # Step 1: Compute L1 norm per feature vector
        # Mean absolute value across the normalized dimension
        l1_norm = x.abs().mean(dim=-1, keepdim=True)  # (..., 1)

        # Step 2: Round to nearest power-of-2 using STE
        # log2(l1_norm) → round → 2^rounded
        log2_norm = torch.log2(l1_norm + self.eps)  # avoid log2(0)
        rounded_log2 = _round_ste(log2_norm)  # STE: forward=round, backward=identity
        power_of_2 = torch.pow(2.0, rounded_log2)  # 2^round(log2(norm))

        # Step 3: Normalize (equivalent to bit-shifting in hardware)
        # x / 2^n is the same as x >> n on integers
        x_normalized = x / (power_of_2 + self.eps)

        # Clamp to prevent explosion when power_of_2 is very small.
        # When l1_norm → 0: log2(eps) ≈ -20, 2^(-20) ≈ 1e-6, x/1e-6 → huge.
        # Clamp bounds the normalized value before gamma scaling.
        x_normalized = torch.clamp(x_normalized, -10.0, 10.0)

        # Step 4: Apply learnable scale
        return x_normalized * self.gamma

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}"


class _RoundSTE(torch.autograd.Function):
    """Round with straight-through estimator for gradient."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # STE: pass gradient through unchanged


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest integer with straight-through estimator."""
    return _RoundSTE.apply(x)


def fold_bspn_into_weights(linear: nn.Linear, bspn: BitShiftPowerNorm) -> nn.Linear:
    """Fold BSPN gamma into preceding Linear layer weights for Lava export.

    After folding: Linear_new(x) ≈ BSPN(Linear_old(x))
    This eliminates the BSPN layer, making it a pure Linear → LIF pipeline.

    Note: This is an approximation. The bit-shift normalization factor depends
    on the input, so only gamma can be exactly folded. The shift normalization
    is handled by the LIF threshold adjustment.

    Args:
        linear: preceding nn.Linear layer (must have bias=False)
        bspn: BitShiftPowerNorm layer to fold

    Returns:
        New nn.Linear with gamma folded into weights
    """
    assert linear.bias is None, "Linear must have bias=False for BSPN folding"

    gamma = bspn.gamma.data  # (out_features,)
    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=False)
    # W_new = diag(gamma) @ W_old → each output row scaled by gamma
    new_linear.weight.data = linear.weight.data * gamma.unsqueeze(1)
    return new_linear
