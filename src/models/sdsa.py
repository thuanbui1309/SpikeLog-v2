"""
SDSA: Spike-Driven Self-Attention — neuromorphic-compatible attention.

Reference: SDT-V2 (ICLR 2024) — Meta Spike-Driven Transformer

Standard attention:
    attn = softmax(Q @ K^T / sqrt(d)) @ V    → requires matmul (NOT neuromorphic)

Spike-Driven attention (SDSA):
    Q, K, V are binary spikes (0/1) from LIF neurons
    attn = AND(Q, K) → popcount → accumulate(V)
    → Only uses: AND, addition, comparison → fully neuromorphic

How it works:
    1. Project input to Q, K, V via Linear (no bias)
    2. Normalize with BSPN (or LayerNorm for ablation)
    3. Pass through LIF → binary spikes
    4. AND(Q_spike, K_spike) → per-position match scores (integer)
    5. Weighted sum of V_spike using match scores
    6. Output projection → continuous float (for training/distillation)

Why neuromorphic-compatible:
    - Q, K, V are binary (0/1) after LIF
    - AND is a single spike operation
    - Accumulation is integer addition
    - No softmax, no division, no matmul of floats

Note on attn_lif / out_lif (NOT used during training):
    SDT-V2 and SpikeBERT have LIF neurons after attention output. However,
    binary attention output is too sparse for anomaly score regression
    (the pairwise score degenerates when output is binary).
    For Lava hardware deployment, attn_lif/out_lif are added during export.
    This follows the train-with-float, deploy-with-spike convention.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode

from src.models.bspn import BitShiftPowerNorm


class SpikeDrivenSelfAttention(nn.Module):
    """Spike-Driven Self-Attention (SDSA).

    Replaces standard matmul attention with AND + accumulate on binary spikes.

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        norm_type: "bspn" (neuromorphic) or "layernorm" (ablation)
        use_bias: whether Linear layers have bias (False for neuromorphic)
        tau: LIF time constant
        v_threshold: LIF firing threshold
    """

    def __init__(self, d_model: int, n_heads: int, norm_type: str = "bspn",
                 use_bias: bool = False, tau: float = 2.0, v_threshold: float = 0.3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k  # for integer scaling (popcount normalization)

        # Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=use_bias)

        # Normalization before LIF (Q, K, V paths)
        if norm_type == "bspn":
            self.q_norm = BitShiftPowerNorm(d_model)
            self.k_norm = BitShiftPowerNorm(d_model)
            self.v_norm = BitShiftPowerNorm(d_model)
        else:
            self.q_norm = nn.LayerNorm(d_model)
            self.k_norm = nn.LayerNorm(d_model)
            self.v_norm = nn.LayerNorm(d_model)

        # LIF neurons → produce binary spikes (Q, K, V paths)
        self.q_lif = LIFNode(tau=tau, v_threshold=v_threshold,
                             surrogate_function=_atan_surrogate(),
                             detach_reset=False)
        self.k_lif = LIFNode(tau=tau, v_threshold=v_threshold,
                             surrogate_function=_atan_surrogate(),
                             detach_reset=False)
        self.v_lif = LIFNode(tau=tau, v_threshold=v_threshold,
                             surrogate_function=_atan_surrogate(),
                             detach_reset=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, L, D) or (B, L, D) — spike tensor from previous layer
               T = timesteps (for temporal coding)

        Returns:
            (B, T, L, D) or (B, L, D) — same shape as input
        """
        has_time = x.dim() == 4
        if has_time:
            B, T, L, D = x.shape
            # Process each timestep
            outputs = []
            for t in range(T):
                out_t = self._forward_single(x[:, t])  # (B, L, D)
                outputs.append(out_t)
            return torch.stack(outputs, dim=1)  # (B, T, L, D)
        else:
            return self._forward_single(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single timestep.

        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape

        # Project → Normalize → LIF → binary spikes
        q = self.q_lif(self.q_norm(self.q_linear(x)))  # (B, L, D) binary
        k = self.k_lif(self.k_norm(self.k_linear(x)))  # (B, L, D) binary
        v = self.v_lif(self.v_norm(self.v_linear(x)))  # (B, L, D) binary

        # Reshape to multi-head: (B, n_heads, L, d_k)
        q = q.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Spike-driven attention: AND(Q, K) + accumulate(V)
        # Q and K are binary spikes → element-wise AND = element-wise multiply
        # This produces integer match scores, NOT floating point matmul
        #
        # Standard: attn = softmax(Q @ K^T / sqrt(d)) @ V
        # SDSA:     score[i,j] = popcount(AND(Q[i], K[j])) = sum(Q[i] * K[j])
        #           out[i] = sum_j(score[i,j] * V[j])
        #
        # Since Q, K are binary (0/1), Q @ K^T gives integer popcount scores
        # Since V is binary (0/1), the weighted sum is also integer
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L) integer

        # No softmax! Just integer scores from popcount
        # Scale by 1/d_k for training stability (folded out in Lava export)
        attn_scores = attn_scores / self.scale

        # Weighted sum of V spikes
        out = torch.matmul(attn_scores, v)  # (B, H, L, d_k)

        # Reshape back + output projection (continuous float for training)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_linear(out)

    # NOTE: No custom .reset() method — spikingjelly's reset_net() handles
    # LIF nodes (MemoryModule) automatically. A custom .reset() on a non-MemoryModule
    # triggers thousands of warnings per epoch, flooding the log file.


def _atan_surrogate():
    """ATan surrogate gradient function for LIF neurons."""
    from spikingjelly.activation_based.surrogate import ATan
    return ATan()
