"""
TEBN: Temporal Effective Batch Normalization for SNNs.

Reference: Duan et al., "Temporal Effective Batch Normalization in Spiking Neural
Networks" (NeurIPS 2022).

Extends standard BN with per-timestep learnable rescaling factors lambda_t.
Shared BN statistics across timesteps, but each timestep has its own scale.

Key advantages over tdBN:
    - Captures temporal dynamics (spike distributions vary across timesteps)
    - More expressive with minimal parameter overhead (one scalar per timestep)
    - Still fully foldable: at inference, BN stats are fixed →
      one set of folded weights per timestep (or averaged for single-pass)

Neuromorphic deployment (fold into weights):
    Same as tdBN: BN(x) → static affine → fold into Linear → fold bias into LIF.
    The per-timestep lambda is absorbed into the gamma scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEffectiveBatchNorm(nn.Module):
    """Temporal Effective Batch Normalization.

    Standard BN statistics shared across timesteps, with per-timestep
    learnable rescaling factor lambda_t.

    For single-timestep input (B, D), behaves identically to standard BN
    with threshold-dependent scaling.

    Args:
        num_features: size of the last dimension (d_model)
        alpha: threshold-dependent scaling (default 1.0, learnable)
        momentum: EMA momentum for running stats
        eps: epsilon for numerical stability
        max_timesteps: maximum number of timesteps for lambda_t params
    """

    def __init__(
        self,
        num_features: int,
        alpha: float = 1.0,
        momentum: float = 0.1,
        eps: float = 1e-5,
        max_timesteps: int = 8,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.max_timesteps = max_timesteps

        # Standard BN affine parameters (shared across timesteps)
        self.weight = nn.Parameter(torch.ones(num_features))   # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))    # beta
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Per-timestep rescaling (TEBN's key contribution)
        # Initialized to 1.0 so initial behavior = standard BN
        self.lambda_t = nn.Parameter(torch.ones(max_timesteps))

        # Running statistics
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
        if x.dim() == 2:
            # No temporal dimension — standard BN with alpha scaling
            return self._bn_forward(x) * self.alpha * self.lambda_t[0]

        B, T, D = x.shape

        # Flatten all timesteps for shared BN statistics
        flat = x.reshape(B * T, D)

        if self.training:
            out = F.batch_norm(
                flat, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=True, momentum=self.momentum, eps=self.eps,
            )
        else:
            out = F.batch_norm(
                flat, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=False, eps=self.eps,
            )

        out = out.reshape(B, T, D)

        # Apply threshold-dependent scaling
        out = out * self.alpha

        # Apply per-timestep rescaling
        # lambda_t: (max_timesteps,) → use first T entries
        lt = self.lambda_t[:T].view(1, T, 1)  # (1, T, 1) broadcast over B, D
        out = out * lt

        return out

    def _bn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard BN forward for 2D input."""
        if self.training:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=True, momentum=self.momentum, eps=self.eps,
            )
        else:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=False, eps=self.eps,
            )

    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, "
                f"alpha={self.alpha.item():.3f}, "
                f"max_timesteps={self.max_timesteps}, "
                f"momentum={self.momentum}, eps={self.eps}")


def fold_tebn_into_linear(linear: nn.Linear, tebn: TemporalEffectiveBatchNorm) -> nn.Linear:
    """Fold TEBN into preceding Linear layer for neuromorphic export.

    Same as tdBN folding: BN stats → static affine → merge with Linear.
    Per-timestep lambda is averaged (single-pass deployment) or the caller
    can fold per-timestep if multi-pass is needed.

    Args:
        linear: preceding Linear layer
        tebn: TemporalEffectiveBatchNorm to fold

    Returns:
        New Linear layer with BN folded in (has bias).
    """
    alpha = tebn.alpha.data
    gamma = tebn.weight.data
    beta = tebn.bias.data
    mu = tebn.running_mean
    var = tebn.running_var
    eps = tebn.eps

    # Average lambda across timesteps for single-pass deployment
    avg_lambda = tebn.lambda_t.data.mean()

    std_inv = 1.0 / torch.sqrt(var + eps)
    w_bn = gamma * std_inv * alpha * avg_lambda
    b_bn = (beta - gamma * mu * std_inv) * alpha * avg_lambda

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
