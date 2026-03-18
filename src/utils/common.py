"""Common utilities: seed, device, config loading, profiling."""

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(config_device: str = "auto") -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def load_config(base_path: str, variant_path: str, dataset_path: str | None = None) -> dict:
    """Load base config, merge with dataset and variant overrides."""
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f)
        base = _deep_merge(base, dataset_cfg)
    with open(variant_path, "r", encoding="utf-8") as f:
        variant = yaml.safe_load(f)
    return _deep_merge(base, variant)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@contextmanager
def timer(label: str = ""):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[Timer] {label}: {elapsed:.2f}s")


class Profiler:
    """Simple profiler to find training bottlenecks."""

    def __init__(self):
        self.records = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed)

    def summary(self) -> dict:
        result = {}
        for name, times in self.records.items():
            result[name] = {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "count": len(times),
            }
        return result

    def print_summary(self):
        print(f"\n{'Phase':<30} {'Total (s)':>10} {'Mean (s)':>10} {'Count':>8}")
        print("-" * 65)
        for name, stats in sorted(self.summary().items(), key=lambda x: -x[1]["total"]):
            print(f"{name:<30} {stats['total']:>10.3f} {stats['mean']:>10.4f} {stats['count']:>8}")

    def get_peak_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0
