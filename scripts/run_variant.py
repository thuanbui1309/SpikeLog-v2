"""
Run SpikeLog-v2 experiments: for each dataset × variant, preprocess → embed → train → predict.
Optionally run Lava simulation + energy profiling.

Usage:
    python scripts/run_variant.py --pending                            # run all pending
    python scripts/run_variant.py --pending --dataset bgl              # pending for BGL
    python scripts/run_variant.py s0_spikelog_baseline --dataset bgl   # specific variant
    python scripts/run_variant.py s2_spikelog_bspn --force --lava      # re-run + energy
    python scripts/run_variant.py --list                               # show all status
"""

import argparse
import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.utils.common import load_config, seed_everything, get_device, Profiler
from src.utils.registry import (
    get_status, get_pending_variants,
    mark_running, mark_completed, mark_failed,
    print_summary,
)


class _Tee:
    """Write to both a stream and a file simultaneously."""
    def __init__(self, stream, filepath: str):
        self._stream = stream
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Required for sys.stdout/stderr compatibility
    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return self._stream.encoding


def _setup_run_logger(log_path: str):
    """Redirect stdout+stderr to both console and log_path."""
    sys.stdout = _Tee(sys.__stdout__, log_path)
    sys.stderr = _Tee(sys.__stderr__, log_path)
    # Also configure Python logging to use the same file
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
               for h in root.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(fh)


def _restore_streams():
    """Restore original stdout/stderr (close tee files)."""
    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, _Tee):
        sys.stderr.close()
        sys.stderr = sys.__stderr__

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
VARIANTS_DIR = CONFIGS_DIR / "variants"
DATASETS_DIR = CONFIGS_DIR / "datasets"


def discover_variants() -> dict[str, Path]:
    """Find all variant config files."""
    import yaml
    variants = {}
    for f in sorted(VARIANTS_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        vid = cfg.get("variant", {}).get("id", f.stem)
        variants[vid] = f
    return variants


def discover_datasets() -> dict[str, Path]:
    """Find all dataset config files."""
    import yaml
    datasets = {}
    for f in sorted(DATASETS_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        name = cfg.get("dataset", {}).get("name", f.stem)
        datasets[name] = f
    return datasets


def ensure_data(config: dict, project_root: str, dataset: str) -> str:
    """Download (if needed) and preprocess the dataset."""
    output_dir = os.path.join(project_root, config["data"]["output_dir"], dataset)
    train_file = os.path.join(output_dir, "train_normal.pkl")

    if os.path.exists(train_file):
        print(f"[✓] Processed data found: {output_dir}")
        return output_dir

    # Check raw log file exists; download if missing
    raw_dir = os.path.join(project_root, config["data"]["raw_dir"], dataset)
    log_file = os.path.join(raw_dir, config["dataset"]["log_file"])
    if not os.path.exists(log_file):
        print(f"[→] Raw data not found for {dataset}. Downloading...")
        from src.data.download import download_dataset
        abs_raw_dir = os.path.join(project_root, config["data"]["raw_dir"])
        download_dataset(config["dataset"], abs_raw_dir, project_root)

    print(f"[→] Preprocessing {dataset}...")
    from src.data.preprocess import prepare_dataset
    prepare_dataset(config, project_root)
    return output_dir


def ensure_embeddings(config: dict, project_root: str, dataset: str) -> str:
    """Check if event vectors exist; generate if not."""
    output_dir = os.path.join(project_root, config["data"]["output_dir"], dataset)
    vectors_file = os.path.join(output_dir, "event_vectors.npy")

    if os.path.exists(vectors_file):
        print(f"[✓] Event vectors found: {vectors_file}")
        return vectors_file

    print(f"[→] Event vectors not found for {dataset}. Generating embeddings...")
    from src.data.embedding import generate_event_vectors
    generate_event_vectors(config, project_root)
    return vectors_file


def run_variant(
    variant_id: str,
    dataset_name: str,
    dataset_path: Path,
    force: bool = False,
    run_lava: bool = False,
):
    """Run a single (variant, dataset) experiment."""
    project_root = str(PROJECT_ROOT)
    variants = discover_variants()

    if variant_id not in variants:
        print(f"Error: variant '{variant_id}' not found.")
        print(f"Available: {', '.join(variants.keys())}")
        return

    if not force:
        status = get_status(project_root, dataset_name, variant_id)
        if status == "completed":
            print(f"[skip] {dataset_name}/{variant_id} already completed. Use --force to re-run.")
            return

    config = load_config(
        str(CONFIGS_DIR / "base.yaml"),
        str(variants[variant_id]),
        str(dataset_path),
    )

    seed_everything(config.get("seed", 42))
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Running: {dataset_name} / {variant_id} ({config['variant']['name']})")
    print(f"Device:  {device}")
    print(f"{'='*60}\n")

    model_dir = os.path.join(project_root, "results", dataset_name, variant_id)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "run.log")

    mark_running(project_root, dataset_name, variant_id, str(variants[variant_id]))
    _setup_run_logger(log_path)
    print(f"Log file: {log_path}")

    profiler = Profiler()
    metrics = {}

    try:
        # Step 1: Preprocess
        with profiler.track("preprocessing"):
            ensure_data(config, project_root, dataset_name)

        # Step 2: Generate embeddings (shared across all variants for this dataset)
        with profiler.track("embeddings"):
            ensure_embeddings(config, project_root, dataset_name)

        # Step 3: Train
        with profiler.track("training"):
            if config.get("distillation", {}).get("enabled", False):
                from src.training.train_distill import train_distill
                train_distill(config, project_root)
            else:
                from src.training.train import train
                train(config, project_root)

        # Step 4: Predict + evaluate
        with profiler.track("prediction"):
            from src.training.predict import predict
            metrics = predict(config, project_root)

        # Step 5: Lava energy profiling (optional, s2/s3 only)
        if run_lava and config.get("lava", {}).get("enabled", False):
            with profiler.track("lava"):
                try:
                    from src.lava.profiler import profile_energy
                    energy_metrics = profile_energy(config, project_root, variant_id)
                    metrics["energy"] = energy_metrics
                    print(f"  Energy ratio: {energy_metrics.get('energy_ratio', 'N/A')}x")
                except ImportError:
                    print("  [warn] lava-nc not installed, skipping Lava simulation")
                except Exception as e:
                    print(f"  [warn] Lava simulation failed: {e}")

        profiler.print_summary()
        metrics["profiling"] = profiler.summary()

        mark_completed(project_root, dataset_name, variant_id, metrics)
        print(f"\n[✓] {dataset_name}/{variant_id} completed!")
        print(f"    F1={metrics.get('f1', 'N/A')}  "
              f"P={metrics.get('precision', 'N/A')}  "
              f"R={metrics.get('recall', 'N/A')}")

    except Exception as e:
        mark_failed(project_root, dataset_name, variant_id, str(e))
        print(f"\n[✗] {dataset_name}/{variant_id} failed: {e}")
        traceback.print_exc()

    finally:
        _restore_streams()


def main():
    parser = argparse.ArgumentParser(description="Run SpikeLog-v2 experiments")
    parser.add_argument("variant", nargs="?", help="Variant ID (e.g., s2_spikelog_bspn)")
    parser.add_argument("--dataset", help="Dataset name (e.g., bgl, hdfs)")
    parser.add_argument("--force", action="store_true", help="Re-run even if completed")
    parser.add_argument("--list", action="store_true", help="Show all variant/dataset status")
    parser.add_argument("--pending", action="store_true", help="Run all pending variants")
    parser.add_argument("--lava", action="store_true", help="Run Lava energy profiling")
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    all_datasets = discover_datasets()
    all_variants = discover_variants()

    # Filter datasets
    if args.dataset:
        if args.dataset not in all_datasets:
            print(f"Error: dataset '{args.dataset}' not found.")
            print(f"Available: {', '.join(all_datasets.keys())}")
            return
        datasets = {args.dataset: all_datasets[args.dataset]}
    else:
        datasets = all_datasets

    if args.list:
        print(f"\nDatasets: {', '.join(all_datasets.keys())}")
        print(f"Variants: {', '.join(all_variants.keys())}\n")
        for ds_name in datasets:
            print(f"--- {ds_name} ---")
            for vid in all_variants:
                status = get_status(project_root, ds_name, vid) or "not started"
                mark = "✓" if status == "completed" else ("→" if status == "running" else "·")
                print(f"  [{mark}] {vid:<35} [{status}]")
            print_summary(project_root, ds_name)
        return

    if args.pending:
        for ds_name, ds_path in datasets.items():
            pending = get_pending_variants(project_root, ds_name, list(all_variants.keys()))
            if not pending:
                print(f"[{ds_name}] All variants completed!")
                print_summary(project_root, ds_name)
                continue

            print(f"\n[{ds_name}] Pending: {', '.join(pending)}")
            for vid in pending:
                run_variant(vid, ds_name, ds_path, force=args.force, run_lava=args.lava)
            print_summary(project_root, ds_name)
        return

    if not args.variant:
        parser.print_help()
        return

    for ds_name, ds_path in datasets.items():
        run_variant(args.variant, ds_name, ds_path, force=args.force, run_lava=args.lava)


if __name__ == "__main__":
    main()
