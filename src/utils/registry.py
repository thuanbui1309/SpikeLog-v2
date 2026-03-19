"""
Experiment registry — tracks which variants have been run and their results.
Supports incremental runs: only execute variants not yet completed.
Organized per-dataset: results/{dataset}/registry.json
"""

import json
import os
from datetime import datetime
from pathlib import Path


def _registry_path(project_root: str, dataset: str) -> str:
    return os.path.join(project_root, "results", dataset, "registry.json")


def _load_registry(project_root: str, dataset: str) -> dict:
    path = _registry_path(project_root, dataset)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"dataset": dataset, "experiments": {}}


def _save_registry(project_root: str, dataset: str, registry: dict):
    path = _registry_path(project_root, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def get_status(project_root: str, dataset: str, variant_id: str) -> str | None:
    """Get status of a variant: None, 'running', 'completed', 'failed'."""
    reg = _load_registry(project_root, dataset)
    entry = reg["experiments"].get(variant_id)
    if entry is None:
        return None
    return entry.get("status")


def get_completed_variants(project_root: str, dataset: str) -> list[str]:
    """Return list of variant IDs that completed successfully."""
    reg = _load_registry(project_root, dataset)
    return [
        vid for vid, entry in reg["experiments"].items()
        if entry.get("status") == "completed"
    ]


def get_pending_variants(project_root: str, dataset: str, all_variant_ids: list[str]) -> list[str]:
    """Return variant IDs that haven't been completed yet."""
    completed = set(get_completed_variants(project_root, dataset))
    return [vid for vid in all_variant_ids if vid not in completed]


def mark_running(project_root: str, dataset: str, variant_id: str, config_path: str):
    """Mark a variant as currently running."""
    reg = _load_registry(project_root, dataset)
    reg["experiments"][variant_id] = {
        "status": "running",
        "config": config_path,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "metrics": {},
    }
    _save_registry(project_root, dataset, reg)


def mark_completed(project_root: str, dataset: str, variant_id: str, metrics: dict):
    """Mark a variant as completed with its metrics."""
    reg = _load_registry(project_root, dataset)
    if variant_id not in reg["experiments"]:
        reg["experiments"][variant_id] = {}
    reg["experiments"][variant_id].update({
        "status": "completed",
        "completed_at": datetime.now().isoformat(),
        "metrics": metrics,
    })
    _save_registry(project_root, dataset, reg)


def mark_failed(project_root: str, dataset: str, variant_id: str, error: str):
    """Mark a variant as failed with error info."""
    reg = _load_registry(project_root, dataset)
    if variant_id not in reg["experiments"]:
        reg["experiments"][variant_id] = {}
    reg["experiments"][variant_id].update({
        "status": "failed",
        "completed_at": datetime.now().isoformat(),
        "error": error,
    })
    _save_registry(project_root, dataset, reg)


def print_summary(project_root: str, dataset: str):
    """Print a summary table of all experiments for a dataset."""
    reg = _load_registry(project_root, dataset)
    experiments = reg.get("experiments", {})

    if not experiments:
        print(f"  No experiments recorded for {dataset}.")
        return

    print(f"\n  [{dataset}]")
    print(f"  {'Variant':<30} {'Status':<12} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Train Time':>12}")
    print(f"  {'-' * 98}")
    for vid, entry in sorted(experiments.items()):
        status = entry.get("status", "?")
        metrics = entry.get("metrics", {})
        acc = f"{metrics.get('accuracy', 0) * 100:.2f}%" if metrics.get("accuracy") else "—"
        f1 = f"{metrics.get('f1', 0) * 100:.2f}%" if metrics.get("f1") else "—"
        prec = f"{metrics.get('precision', 0) * 100:.2f}%" if metrics.get("precision") else "—"
        recall = f"{metrics.get('recall', 0) * 100:.2f}%" if metrics.get("recall") else "—"
        train_time = metrics.get("train_time", "—")
        print(f"  {vid:<30} {status:<12} {acc:>8} {f1:>8} {prec:>8} {recall:>8} {str(train_time):>12}")
    print()


def print_all_summaries(project_root: str):
    """Print summary tables for all datasets that have results."""
    results_dir = os.path.join(project_root, "results")
    if not os.path.exists(results_dir):
        print("No results yet.")
        return

    datasets = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and os.path.exists(os.path.join(results_dir, d, "registry.json"))
    ])

    if not datasets:
        print("No results yet.")
        return

    for ds in datasets:
        print_summary(project_root, ds)
