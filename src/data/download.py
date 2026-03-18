"""Download datasets for log anomaly detection."""

import os
import shutil
import tarfile
import urllib.request
import zipfile


def download_dataset(ds_cfg: dict, raw_dir: str, project_root: str = "."):
    """Download or copy a dataset based on its config."""
    name = ds_cfg["name"]
    output_dir = os.path.join(raw_dir, name)
    log_file = os.path.join(output_dir, ds_cfg["log_file"])

    if os.path.exists(log_file):
        print(f"[✓] {name} already downloaded")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Bundled datasets: copy from data/bundled/ instead of downloading
    if ds_cfg.get("bundled"):
        bundled_dir = os.path.join(project_root, "data", "bundled", name)
        if not os.path.exists(bundled_dir):
            raise FileNotFoundError(
                f"Bundled data not found: {bundled_dir}. "
                f"Ensure data/bundled/{name}/ exists in the project."
            )
        print(f"    Copying bundled {name} data...")
        for f in os.listdir(bundled_dir):
            src = os.path.join(bundled_dir, f)
            dst = os.path.join(output_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        with open(log_file) as f:
            lines = sum(1 for _ in f)
        print(f"    {ds_cfg['log_file']}: {lines} lines")
        return

    # Download from URL
    url = ds_cfg["download_url"]
    extract_type = ds_cfg.get("download_extract", "zip")

    is_tar = extract_type in ("tar", "tar.gz")
    ext = ".tar.gz" if is_tar else ".zip"
    archive_path = os.path.join(output_dir, f"{name}{ext}")
    if not os.path.exists(archive_path):
        print(f"    Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, archive_path)
        print("    Download complete.")

    # Extract
    print("    Extracting...")
    if is_tar:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)
    else:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)

    # Move files to top-level if nested
    expected_log = os.path.join(output_dir, ds_cfg["log_file"])
    if not os.path.exists(expected_log):
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f == ds_cfg["log_file"]:
                    src = os.path.join(root, f)
                    os.rename(src, expected_log)
                # Also move label files if present
                label_file = ds_cfg.get("label_file")
                if label_file and f == label_file:
                    src = os.path.join(root, f)
                    dest = os.path.join(output_dir, label_file)
                    if src != dest:
                        os.rename(src, dest)

    if os.path.exists(expected_log):
        with open(expected_log) as f:
            lines = sum(1 for _ in f)
        print(f"    {ds_cfg['log_file']}: {lines} lines")
    else:
        print(f"    Warning: {ds_cfg['log_file']} not found in {output_dir}")
        print(f"    Download manually from https://github.com/logpai/loghub")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    with open(args.dataset_config) as f:
        cfg = yaml.safe_load(f)
    download_dataset(cfg["dataset"], args.raw_dir)
