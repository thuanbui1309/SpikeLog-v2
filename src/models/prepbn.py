"""
PRepBN: Progressive Re-parameterized Batch Normalization.

Reference: Guo et al., "SLAB: Efficient Transformers with Simplified Linear
Attention and Progressive Re-parameterized Batch Normalization" (ICML 2024).
Also: Huang et al., "Decision SpikeFormer" (CVPR 2025) for spiking transformers.

Key idea: train starting as LayerNorm, progressively transition to BatchNorm.
    Norm(x) = (1 - alpha_t) * LN(x) + alpha_t * BN(x)
    alpha_t linearly increases from 0 → 1 over training.
    At the end of training: pure BN → foldable → neuromorphic.

Advantages:
    - No KD needed: model self-transitions from LN (good accuracy) to BN (deployable)
    - Smooth optimization landscape throughout training
    - Validated for spiking transformers (Decision SpikeFormer, CVPR 2025)

Neuromorphic deployment:
    After training, alpha_t = 1.0 → pure BN → fold into Linear weights (same as tdBN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveBatchNorm(nn.Module):
    """Progressive LN→BN transition for SNN normalization.

    During training, interpolates between LayerNorm and BatchNorm:
        output = (1 - mix) * LN(x) + mix * BN(x)
    where mix increases from 0 → 1 over the transition period.

    After training (or after transition completes), behaves as pure BN.

    Args:
        num_features: size of the last dimension (d_model)
        alpha: threshold-dependent scaling for BN (learnable)
        momentum: EMA momentum for BN running stats
        eps: epsilon for numerical stability
        transition_epochs: number of epochs for full LN→BN transition
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        momentum: float = 0.1,
        eps: float = 1e-5,
        transition_epochs: int = 20,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.transition_epochs = transition_epochs

        # BN parameters
        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # LN parameters (separate from BN)
        self.ln = nn.LayerNorm(num_features)

        # BN running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # Mixing coefficient: 0 = pure LN, 1 = pure BN
        # Not a parameter — set externally by training loop
        self.register_buffer('mix_ratio', torch.tensor(0.0))

    def set_epoch(self, epoch: int):
        """Update mix_ratio based on current epoch.

        Call this at the start of each epoch in the training loop.
        """
        if self.transition_epochs <= 0:
            ratio = 1.0
        else:
            ratio = min(1.0, epoch / self.transition_epochs)
        self.mix_ratio.fill_(ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) or (B, D)
        Returns:
            normalized x, same shape
        """
        mix = self.mix_ratio.item()

        # Pure BN mode (after transition or during eval)
        if mix >= 1.0 or not self.training:
            return self._bn_forward(x) * self.alpha

        # Pure LN mode (start of training)
        if mix <= 0.0:
            return self.ln(x)

        # Progressive interpolation
        ln_out = self.ln(x)
        bn_out = self._bn_forward(x) * self.alpha
        return (1 - mix) * ln_out + mix * bn_out

    def _bn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard BN forward, handles both 2D and 3D."""
        if x.dim() == 3:
            B, T, D = x.shape
            flat = x.reshape(B * T, D)
        else:
            flat = x

        if self.training:
            out = F.batch_norm(
                flat, self.running_mean, self.running_var,
                self.bn_weight, self.bn_bias,
                training=True, momentum=self.momentum, eps=self.eps,
            )
        else:
            out = F.batch_norm(
                flat, self.running_mean, self.running_var,
                self.bn_weight, self.bn_bias,
                training=False, eps=self.eps,
            )

        if x.dim() == 3:
            out = out.reshape(B, T, D)
        return out

    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, "
                f"transition_epochs={self.transition_epochs}, "
                f"mix_ratio={self.mix_ratio.item():.2f}, "
                f"alpha={self.alpha.item():.3f}")


def fold_prepbn_into_linear(linear: nn.Linear, prepbn: ProgressiveBatchNorm) -> nn.Linear:
    """Fold PRepBN (in pure-BN mode) into preceding Linear for neuromorphic export.

    Only valid after training is complete (mix_ratio = 1.0, pure BN mode).
    Same folding math as tdBN.
    """
    alpha = prepbn.alpha.data
    gamma = prepbn.bn_weight.data
    beta = prepbn.bn_bias.data
    mu = prepbn.running_mean
    var = prepbn.running_var
    eps = prepbn.eps

    std_inv = 1.0 / torch.sqrt(var + eps)
    w_bn = gamma * std_inv * alpha
    b_bn = (beta - gamma * mu * std_inv) * alpha

    W = linear.weight.data
    W_new = W * w_bn.unsqueeze(1)

    if linear.bias is not None:
        b_new = w_bn * linear.bias.data + b_bn
    else:
        b_new = b_bn

    new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
    new_linear.weight.data = W_new
    new_linear.bias.data = b_new
    return new_linear
