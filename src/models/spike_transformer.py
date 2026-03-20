"""
s1/s2/s3: SDSA-enhanced SpikeLog model.

Replaces SpikeLog's flat LIF stack with SDSA blocks that attend across log events,
adding the cross-event context that SpikeLog lacks.

Architecture (SpikeTransformerNet):
    input: (B, seq_len, emb_dim=300)
    → Linear(emb_dim → d_model)         # input projection
    → [SDSA block] × n_layers           # cross-event attention
    → mean pool over seq_len            # (B, d_model)
    output: (B, d_model)

DualSpikeTransformer:
    shared SpikeTransformerNet for both sequences
    → concat([repr1, repr2])            # (B, 2*d_model)
    → Linear(2*d_model → 1)             # anomaly score

Token Pruning (s3):
    After each SDSA block, keep top-K tokens by spike firing rate.
    CLS token always preserved (index 0 if we add it, or we don't use CLS).
    Pruning is active during both training and inference (TP-Spikformer).
    Pad back to original length after pruning to preserve shape.

Reference:
- SDSA: SDT-V2 (ICLR 2024)
- BSPN: Sorbet (ICML 2025)
- Token Pruning: TP-Spikformer (arXiv 2603.00527)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional as sj_functional
from spikingjelly.activation_based.neuron import LIFNode
from src.models.sdsa import SpikeDrivenSelfAttention


class SDSABlock(nn.Module):
    """One Transformer-style block with SDSA + FFN.

    Structure:
        x → Norm → SDSA → + x  (pre-norm residual)
        x → Norm → FFN  → + x
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm_type: str = "bspn",
        use_bias: bool = False,
        tau: float = 2.0,
        v_threshold: float = 0.3,
        ffn_ratio: float = 4.0,
    ):
        super().__init__()
        # SDSA has internal norms (q_norm, k_norm, v_norm) — no external norm1 needed.
        # External norm2 only for FFN path.
        self.attn = SpikeDrivenSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            norm_type=norm_type,
            use_bias=use_bias,
            tau=tau,
            v_threshold=v_threshold,
        )
        self.norm2 = _make_norm(norm_type, d_model)
        ffn_dim = int(d_model * ffn_ratio)
        # LIF replaces GELU: keeps FFN fully spiking for neuromorphic compatibility.
        # reset_net() from spikingjelly traverses Sequential and resets this LIF too.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=use_bias),
            LIFNode(tau=tau, v_threshold=v_threshold,
                    surrogate_function=_atan_surrogate(), detach_reset=False),
            nn.Linear(ffn_dim, d_model, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)
        """
        x = x + self.attn(x)           # SDSA handles norm internally
        x = x + self.ffn(self.norm2(x))
        return x


class TokenPruner(nn.Module):
    """Dynamic token pruning by spike firing rate (TP-Spikformer).

    Scores each token by L1 norm of its representation (proxy for spike activity).
    Keeps top-K tokens, pads rest with zeros. CLS-free: all tokens treated equally.

    Active during both training and inference (model learns sparse patterns).
    """

    def __init__(self, keep_ratio: float = 0.8):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            x_pruned: (B, T, D) — low-score tokens zeroed out (padded)
        """
        B, T, D = x.shape
        k = max(1, int(T * self.keep_ratio))

        # Score each token by L1 norm (spike firing rate proxy)
        scores = x.abs().sum(dim=-1)  # (B, T)

        # Top-K indices per sample
        _, topk_idx = scores.topk(k, dim=1)  # (B, K)

        # Create mask: 1 for kept tokens, 0 for pruned
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, True)  # (B, T)

        # Zero out pruned tokens (pad back to original shape)
        return x * mask.unsqueeze(-1).float()


class SpikeTransformerNet(nn.Module):
    """SDSA-based sequence encoder for s1/s2/s3.

    Replaces SpikeNet's flat LIF stack with attention blocks.

    Args:
        emb_dim   : input embedding dimension (300 for TF-IDF vectors)
        d_model   : internal model dimension
        n_layers  : number of SDSA blocks
        n_heads   : attention heads
        norm_type : "layernorm" (s1) or "bspn" (s2/s3)
        use_bias  : False for neuromorphic (s2/s3), True for ablation (s1)
        prune_after_layers : list of layer indices (1-indexed) after which to prune
        keep_ratio         : fraction of tokens to keep per pruning step
    """

    def __init__(
        self,
        emb_dim: int = 300,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        norm_type: str = "bspn",
        use_bias: bool = False,
        tau: float = 2.0,
        v_threshold: float = 0.3,
        prune_after_layers: list[int] | None = None,
        keep_ratio: float = 0.8,
    ):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model, bias=use_bias)

        self.blocks = nn.ModuleList([
            SDSABlock(d_model, n_heads, norm_type, use_bias, tau, v_threshold)
            for _ in range(n_layers)
        ])

        self.prune_after = set(prune_after_layers or [])
        self.pruner = TokenPruner(keep_ratio) if self.prune_after else None
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, emb_dim)
        Returns:
            repr: (B, d_model) — mean-pooled representation
        """
        x = self.input_proj(x)  # (B, T, d_model)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) in self.prune_after and self.pruner is not None:
                x = self.pruner(x)

        # Mean pooling (ignore zero-padded tokens for pruning)
        # Use non-zero mask to avoid counting pruned tokens in mean
        mask = (x.abs().sum(dim=-1) > 0).float()  # (B, T)
        n = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        repr = (x * mask.unsqueeze(-1)).sum(dim=1) / n   # (B, d_model)
        return repr


class DualSpikeTransformer(nn.Module):
    """s1/s2/s3: Dual pairwise SNN with SDSA attention.

    Shared SpikeTransformerNet encodes both sequences independently,
    concatenated representation scored by a Linear head.

    Input:
        x1, x2: (B, T, emb_dim)
    Output:
        score: (B, 1)  — higher = more anomalous pair
    """

    def __init__(
        self,
        emb_dim: int = 300,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        norm_type: str = "bspn",
        use_bias: bool = False,
        tau: float = 2.0,
        v_threshold: float = 0.3,
        prune_after_layers: list[int] | None = None,
        keep_ratio: float = 0.8,
    ):
        super().__init__()
        self.encoder = SpikeTransformerNet(
            emb_dim=emb_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            norm_type=norm_type,
            use_bias=use_bias,
            tau=tau,
            v_threshold=v_threshold,
            prune_after_layers=prune_after_layers,
            keep_ratio=keep_ratio,
        )
        self.output_layer = nn.Linear(d_model * 2, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        sj_functional.reset_net(self.encoder)   # clean start
        r1 = self.encoder(x1)                   # (B, d_model)
        sj_functional.reset_net(self.encoder)   # reset LIF state between x1, x2
        r2 = self.encoder(x2)                   # (B, d_model)
        return self.output_layer(torch.cat([r1, r2], dim=1))  # (B, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Single-sequence encoding for anomaly scoring."""
        return self.encoder(x)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atan_surrogate():
    from spikingjelly.activation_based.surrogate import ATan
    return ATan()


def _make_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "bspn":
        from src.models.bspn import BitShiftPowerNorm
        return BitShiftPowerNorm(d_model)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r} (expected 'bspn' or 'layernorm')")
