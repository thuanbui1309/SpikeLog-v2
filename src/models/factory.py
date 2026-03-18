"""
Model factory: maps variant config → model instance.

Variant architecture identifiers (from configs/variants/*.yaml):
    spikelog_original      → s0: DualSpikeNet (LIF stack + LSTM)
    spikelog_sdsa          → s1/s2: DualSpikeTransformer (SDSA + LayerNorm or BSPN)
    spikelog_sdsa_pruned   → s3: DualSpikeTransformer with token pruning
"""

import torch.nn as nn


def create_model(config: dict) -> nn.Module:
    """Create the appropriate model from the merged variant config.

    Args:
        config: merged config dict (base + dataset + variant)

    Returns:
        model: nn.Module ready for training
    """
    model_cfg = config["model"]
    arch = model_cfg["architecture"]

    if arch == "spikelog_original":
        return _create_spikenet(model_cfg)
    elif arch in ("spikelog_sdsa", "spikelog_sdsa_pruned"):
        return _create_spike_transformer(model_cfg, arch)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")


def _create_spikenet(cfg: dict) -> nn.Module:
    from src.models.spikenet import DualSpikeNet
    return DualSpikeNet(
        num_inputs=cfg.get("embedding_dim", 300),
        num_hidden=cfg.get("hidden", 128),
        num_out=cfg.get("num_out", 32),
        tau=cfg.get("tau", 2.0),
        v_threshold=cfg.get("v_threshold", 1.0),
        out_threshold=cfg.get("out_threshold", 0.1),
        detach_reset=cfg.get("detach_reset", True),
    )


def _create_spike_transformer(cfg: dict, arch: str) -> nn.Module:
    from src.models.spike_transformer import DualSpikeTransformer

    prune_after = cfg.get("prune_after_layers", None) if arch == "spikelog_sdsa_pruned" else None

    return DualSpikeTransformer(
        emb_dim=cfg.get("embedding_dim", 300),
        d_model=cfg.get("hidden", 128),
        n_layers=cfg.get("layers", 2),
        n_heads=cfg.get("attn_heads", 4),
        norm_type=cfg.get("norm", "bspn"),
        use_bias=cfg.get("bias", False),
        tau=cfg.get("tau", 10.0),
        v_threshold=cfg.get("v_threshold", 1.0),
        prune_after_layers=prune_after,
        keep_ratio=cfg.get("keep_ratio", 0.8),
    )
