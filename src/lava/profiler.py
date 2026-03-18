"""
Energy profiler for SpikeLog-v2 (spike-counting mode).

Counts spikes during PyTorch inference, then maps to Loihi 2 energy specs.
Same methodology as SorLog/src/lava/profiler.py but adapted for pairwise SNN.

Loihi 2 energy specs (from Intel):
    - SOP (synapse operation): 0.052 pJ
    - Neuron update: 0.081 pJ per neuron per timestep
    - Spike routing: 0.025 pJ per spike routed

ANN baseline: DualSpikeNet (s0 ANN-equivalent), FMA @ 4.6 pJ (45nm CMOS, SpikeBERT)
"""

import os
import json
from collections import defaultdict

import torch
import numpy as np


def profile_energy(config: dict, project_root: str, variant_id: str) -> dict:
    """Profile energy for the trained model via spike counting.

    Returns dict with energy metrics: energy_ratio, firing_rate, etc.
    """
    from src.models.factory import create_model
    from src.data.embedding import load_event_vectors
    from src.data.dataset import PairwiseTestDataset, collate_test
    from src.utils.common import get_device
    from torch.utils.data import DataLoader

    device = get_device()
    data_cfg = config["data"]
    dataset = config["dataset"]["name"]
    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    model_dir = os.path.join(project_root, "results", dataset, variant_id)
    best_path = os.path.join(model_dir, "best_model.pth")

    event_vectors = load_event_vectors(config, project_root)
    max_seq_len = data_cfg.get("window_size", 100)
    n_cmp = data_cfg.get("n_comparisons", 30)

    # Profile on a small subset
    test_ds = PairwiseTestDataset(
        os.path.join(output_dir, "test.pkl"),
        os.path.join(output_dir, "train_normal.pkl"),
        os.path.join(output_dir, "train_anomaly.pkl"),
        event_vectors, n_cmp, max_seq_len,
    )
    n_profile = min(100, len(test_ds))
    from torch.utils.data import Subset
    profile_ds = Subset(test_ds, list(range(n_profile)))
    loader = DataLoader(profile_ds, batch_size=16, collate_fn=collate_test)

    model = create_model(config).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Energy specs
    sop_pj = 0.052
    neuron_pj = 0.081
    routing_pj = 0.025
    fma_pj = 4.6  # ANN baseline

    spike_stats = _count_spikes(model, loader, device)

    n_samples = spike_stats["n_samples"]
    total_sops = spike_stats["total_sops"]
    total_spikes = spike_stats["total_spikes_routed"]
    total_neurons = spike_stats["total_neurons"]

    energy_pj = (total_sops * sop_pj + total_neurons * neuron_pj +
                 total_spikes * routing_pj)
    energy_per_sample_pj = energy_pj / max(n_samples, 1)

    # ANN comparison: equivalent dense forward pass
    ann_flops = _estimate_ann_flops(config)
    ann_energy_pj = ann_flops * fma_pj
    energy_ratio = ann_energy_pj / energy_per_sample_pj if energy_per_sample_pj > 0 else 0

    result = {
        "mode": "spike_counting",
        "n_samples": n_samples,
        "total_sops": total_sops,
        "energy_per_sample_pj": round(energy_per_sample_pj, 2),
        "energy_per_sample_uj": round(energy_per_sample_pj / 1e6, 4),
        "ann_energy_pj": round(ann_energy_pj, 2),
        "energy_ratio": round(energy_ratio, 1),
        "avg_firing_rate": spike_stats["avg_firing_rate"],
        "per_layer_firing_rate": spike_stats["per_layer_firing_rate"],
    }

    with open(os.path.join(model_dir, "energy_profile.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Energy Profile ({variant_id}):")
    print(f"    Energy/inference: {energy_per_sample_pj/1e6:.4f} µJ")
    print(f"    ANN baseline:     {ann_energy_pj/1e6:.4f} µJ")
    print(f"    Energy savings:   {energy_ratio:.1f}x")
    print(f"    Avg firing rate:  {spike_stats['avg_firing_rate']:.4f}")

    return result


@torch.no_grad()
def _count_spikes(model, loader, device) -> dict:
    from spikingjelly.activation_based.neuron import LIFNode

    total_spikes = 0
    total_possible = 0
    total_spikes_routed = 0
    per_layer_spikes = defaultdict(int)
    per_layer_possible = defaultdict(int)
    n_samples = 0
    spike_counts = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                spike_counts[name] = output.detach()
        return hook

    hooks = []
    lif_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, LIFNode):
            h = module.register_forward_hook(make_hook(f"lif_{lif_idx}_{name}"))
            hooks.append(h)
            lif_idx += 1

    for x_test, x_anom, x_norm, labels in loader:
        x_test = x_test.to(device)
        x_anom = x_anom.to(device)
        B, n_cmp, ref_T, D = x_anom.shape
        T = x_test.size(1)

        spike_counts.clear()
        x_test_exp = x_test.unsqueeze(1).expand(B, n_cmp, T, D).reshape(B * n_cmp, T, D)
        x_anom_flat = x_anom.reshape(B * n_cmp, ref_T, D)
        model(x_test_exp, x_anom_flat)
        n_samples += B

        for name, spikes in spike_counts.items():
            n_s = spikes.sum().item()
            n_p = spikes.numel()
            total_spikes += n_s
            total_possible += n_p
            total_spikes_routed += n_s
            parts = name.split("_")
            layer_key = parts[1] if len(parts) >= 2 else "0"
            per_layer_spikes[layer_key] += n_s
            per_layer_possible[layer_key] += n_p

    for h in hooks:
        h.remove()

    avg_firing_rate = total_spikes / max(total_possible, 1)
    per_layer_rate = {
        f"layer_{k}": round(per_layer_spikes[k] / max(per_layer_possible[k], 1), 4)
        for k in sorted(per_layer_spikes)
    }

    # Rough SOP estimate (spike × avg fan-out)
    d_model = 128
    avg_fan_out = d_model  # approximate
    total_sops = int(total_spikes * avg_fan_out)

    # Neuron count estimate
    n_lif = lif_idx
    total_neurons = n_lif * d_model * n_samples

    return {
        "total_sops": total_sops,
        "total_neurons": total_neurons,
        "total_spikes_routed": total_spikes_routed,
        "n_samples": n_samples,
        "avg_firing_rate": round(avg_firing_rate, 4),
        "per_layer_firing_rate": per_layer_rate,
    }


def _estimate_ann_flops(config) -> float:
    """Estimate equivalent ANN FLOPs for comparison."""
    model_cfg = config["model"]
    seq_len = config["data"].get("window_size", 100)
    emb_dim = model_cfg.get("embedding_dim", 300)
    d_model = model_cfg.get("hidden", 128)
    n_layers = model_cfg.get("layers", 2)
    d_ff = d_model * 4

    # Input projection
    proj = seq_len * emb_dim * d_model * 2
    # Per SDSA block: QKV + attention + FFN
    qkv = 3 * seq_len * d_model * d_model * 2
    attn = seq_len * seq_len * d_model * 2
    ffn = seq_len * d_model * d_ff * 2 * 2
    per_block = qkv + attn + ffn
    # Dual network (2×) + output head
    total = (proj + per_block * n_layers) * 2 + d_model * 2

    return float(total)
