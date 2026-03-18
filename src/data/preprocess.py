"""
Data preprocessing pipeline for SpikeLog-v2.

Key difference from SorLog:
- Preserves labels alongside sequences (needed for pairwise supervised training)
- Outputs train_normal.pkl / train_anomaly.pkl / test.pkl (not masked LM format)
- train_anomaly is the "known anomaly" set used for pairwise training

Pipeline:
    raw log → Drain parser → event sequences → session grouping
    → label extraction → train/test split → save pkl files
"""

import os
import re
import json
import pickle

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def prepare_dataset(config: dict, project_root: str):
    """Full data preparation pipeline for SpikeLog-v2."""
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    dataset = ds_cfg["name"]

    raw_dir = os.path.join(project_root, data_cfg["raw_dir"], dataset)
    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    os.makedirs(output_dir, exist_ok=True)

    log_file = ds_cfg["log_file"]
    log_format = ds_cfg["log_format"]

    sample_lines = ds_cfg.get("sample_lines")
    if sample_lines:
        actual_log_file = f"{os.path.splitext(log_file)[0]}_{sample_lines // 10 ** 6}M{os.path.splitext(log_file)[1]}"
    else:
        actual_log_file = log_file

    # Step 1: Drain parsing
    structured_file = os.path.join(output_dir, actual_log_file + "_structured.csv")
    templates_file = os.path.join(output_dir, actual_log_file + "_templates.csv")
    if not os.path.exists(structured_file):
        print("[1/4] Parsing raw logs with Drain...")
        _run_drain_parser(raw_dir, output_dir, log_file, log_format, ds_cfg)
    else:
        print("[1/4] Structured log already exists, skipping.")

    # Step 2: Event template mapping
    mapping_file = os.path.join(output_dir, "log_templates.json")
    if not os.path.exists(mapping_file):
        print("[2/4] Creating event template mapping...")
        _create_mapping(templates_file, mapping_file)
    else:
        print("[2/4] Template mapping already exists.")

    # Step 3: Session grouping with labels
    sessions_file = os.path.join(output_dir, "sessions.pkl")
    if not os.path.exists(sessions_file):
        print("[3/4] Grouping logs into labeled sessions...")
        _create_labeled_sessions(structured_file, mapping_file, sessions_file, raw_dir, ds_cfg)
    else:
        print("[3/4] Sessions already exist.")

    # Step 4: Train/test split
    train_normal_file = os.path.join(output_dir, "train_normal.pkl")
    if not os.path.exists(train_normal_file):
        print("[4/4] Splitting train/test...")
        _split_train_test(sessions_file, output_dir, data_cfg)
    else:
        print("[4/4] Train/test split already exists.")

    # Save template text file for embedding generation
    template_text_file = os.path.join(output_dir, "templates_text.json")
    if not os.path.exists(template_text_file):
        print("[+] Saving template text for embedding generation...")
        _save_template_texts(templates_file, mapping_file, template_text_file)
    else:
        print("[+] Template texts already saved.")

    print(f"\n[✓] Dataset prepared in {output_dir}")


def _run_drain_parser(input_dir, output_dir, log_file, log_format, ds_cfg):
    try:
        from logparser import Drain
    except ImportError:
        raise ImportError("logparser not found. Install: pip install logparser3")

    sample_lines = ds_cfg.get("sample_lines")
    actual_log_file = log_file
    if sample_lines:
        sampled_name = f"{os.path.splitext(log_file)[0]}_{sample_lines // 10 ** 6}M{os.path.splitext(log_file)[1]}"
        sampled_path = os.path.join(input_dir, sampled_name)
        if not os.path.exists(sampled_path):
            print(f"  Sampling first {sample_lines:,} lines from {log_file}...")
            with open(os.path.join(input_dir, log_file), 'r', errors='ignore') as fin, \
                 open(sampled_path, 'w') as fout:
                for i, line in enumerate(fin):
                    if i >= sample_lines:
                        break
                    fout.write(line)
        actual_log_file = sampled_name

    parser = Drain.LogParser(
        log_format, indir=input_dir, outdir=output_dir,
        depth=ds_cfg.get("drain_depth", 5),
        st=ds_cfg.get("drain_st", 0.5),
        rex=ds_cfg.get("drain_regex", []),
        keep_para=False,
        maxChild=ds_cfg.get("drain_max_child", 100)
    )
    parser.parse(actual_log_file)


def _create_mapping(templates_file, mapping_file):
    log_temp = pd.read_csv(templates_file)
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx + 1 for idx, event in enumerate(log_temp["EventId"])}
    with open(mapping_file, "w") as f:
        json.dump(log_temp_dict, f)
    print(f"  Found {len(log_temp_dict)} event templates")


def _create_labeled_sessions(structured_file, mapping_file, sessions_file, raw_dir, ds_cfg):
    """Group logs into sessions and assign binary labels (0=normal, 1=anomaly)."""
    df = pd.read_csv(structured_file, engine="c", na_filter=False, memory_map=True,
                     dtype={"Date": object, "Time": object})

    with open(mapping_file, "r") as f:
        event_num = json.load(f)

    df["EventIdx"] = df["EventId"].apply(lambda x: event_num.get(x, 0))

    session_type = ds_cfg.get("session_type", "block_id")
    anomaly_labels = ds_cfg.get("anomaly_labels", "per_line")

    if session_type == "block_id":
        sessions = _sessions_block_id(df, raw_dir, ds_cfg)
    elif session_type in ("sliding_window", "fixed_window"):
        sessions = _sessions_window(df, ds_cfg)
    else:
        raise ValueError(f"Unknown session_type: {session_type}")

    print(f"  Total sessions: {len(sessions)}")
    n_anomaly = sum(1 for s in sessions if s[1] == 1)
    print(f"  Normal: {len(sessions) - n_anomaly}, Anomaly: {n_anomaly}")

    with open(sessions_file, "wb") as f:
        pickle.dump(sessions, f)


def _sessions_block_id(df, raw_dir, ds_cfg):
    """HDFS: group by BlockId from Content, label from anomaly_label.csv."""
    # Group events by BlockId
    data_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping by BlockId"):
        blk_ids = re.findall(r'(blk_-?\d+)', str(row['Content']))
        for blk_id in set(blk_ids):
            data_dict[blk_id].append(int(row["EventIdx"]))

    # Load labels
    label_file = os.path.join(raw_dir, ds_cfg.get("label_file", "anomaly_label.csv"))
    label_dict = {}
    if os.path.exists(label_file):
        blk_df = pd.read_csv(label_file)
        for _, row in blk_df.iterrows():
            label_dict[row["BlockId"]] = 1 if str(row["Label"]).strip() == "Anomaly" else 0

    sessions = []
    for blk_id, seq in data_dict.items():
        label = label_dict.get(blk_id, 0)
        sessions.append((seq, label))
    return sessions


def _sessions_window(df, ds_cfg):
    """BGL/Thunderbird: sliding window sessions with per-line labels."""
    # Parse timestamps
    timestamps = _parse_timestamps(df)
    df = df.copy()
    df["_ts"] = timestamps
    df["_is_anomaly"] = df["Label"].apply(lambda x: 0 if str(x).strip() == "-" else 1)

    session_type = ds_cfg.get("session_type", "sliding_window")
    if session_type == "fixed_window":
        window_size = ds_cfg.get("fixed_window_size", 100)
        step_size = ds_cfg.get("fixed_step_size", 100)
        sessions = []
        for start in range(0, len(df), step_size):
            end = min(start + window_size, len(df))
            window = df.iloc[start:end]
            seq = window["EventIdx"].tolist()
            label = 1 if window["_is_anomaly"].sum() > 0 else 0
            sessions.append((seq, label))
        return sessions

    # sliding_window
    window_min = ds_cfg.get("window_size_minutes", 5)
    step_min = ds_cfg.get("step_size_minutes", 5)
    window_td = pd.Timedelta(minutes=window_min)
    step_td = pd.Timedelta(minutes=step_min)

    start_time = df["_ts"].min()
    end_time = df["_ts"].max()

    sessions = []
    t = start_time
    while t < end_time:
        mask = (df["_ts"] >= t) & (df["_ts"] < t + window_td)
        window = df[mask]
        if len(window) > 0:
            seq = window["EventIdx"].tolist()
            label = 1 if window["_is_anomaly"].sum() > 0 else 0
            sessions.append((seq, label))
        t += step_td
    return sessions


def _parse_timestamps(df):
    """Try multiple strategies to parse timestamps."""
    # BGL: Time column 'YYYY-MM-DD-HH.MM.SS.ffffff'
    if "Time" in df.columns:
        try:
            ts = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
            if ts.notna().sum() > len(ts) * 0.5:
                return ts
        except Exception:
            pass

    # Thunderbird: Date + Time combined
    if "Date" in df.columns and "Time" in df.columns:
        try:
            ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                errors="coerce", format="mixed")
            if ts.notna().sum() > len(ts) * 0.5:
                return ts
        except Exception:
            pass

    # Fallback: row index as pseudo-timestamp
    print("  Warning: could not parse timestamps, using row-index windowing")
    return pd.to_datetime(pd.RangeIndex(len(df)), unit="s")


def _split_train_test(sessions_file, output_dir, data_cfg):
    """Split sessions into train_normal, train_anomaly, test."""
    with open(sessions_file, "rb") as f:
        sessions = pickle.load(f)

    normal = [(seq, 0) for seq, label in sessions if label == 0]
    anomaly = [(seq, 1) for seq, label in sessions if label == 1]

    # Shuffle
    rng = np.random.RandomState(42)
    normal_idx = rng.permutation(len(normal)).tolist()
    normal = [normal[i] for i in normal_idx]

    train_ratio = data_cfg.get("train_ratio", 0.7)
    if train_ratio >= 1:
        n_train_normal = int(train_ratio)
    else:
        n_train_normal = int(len(normal) * train_ratio)

    n_train_normal = min(n_train_normal, len(normal))

    # Train: normal sessions + known anomaly sessions (for pairwise)
    # We keep all anomalies accessible during training (weakly supervised)
    # But only n_train_normal sessions appear in train_normal
    train_normal = [seq for seq, _ in normal[:n_train_normal]]
    train_anomaly = [seq for seq, _ in anomaly]

    # Test: remaining normal + all anomaly
    test = [(seq, 0) for seq, _ in normal[n_train_normal:]] + [(seq, 1) for seq, _ in anomaly]

    print(f"  train_normal: {len(train_normal)}")
    print(f"  train_anomaly: {len(train_anomaly)}")
    print(f"  test: {len(test)} ({sum(1 for _, l in test if l == 1)} anomalous)")

    with open(os.path.join(output_dir, "train_normal.pkl"), "wb") as f:
        pickle.dump(train_normal, f)
    with open(os.path.join(output_dir, "train_anomaly.pkl"), "wb") as f:
        pickle.dump(train_anomaly, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test, f)


def _save_template_texts(templates_file, mapping_file, template_text_file):
    """Save EventIdx → EventTemplate text mapping for Word2Vec embedding."""
    with open(mapping_file, "r") as f:
        event_id_to_idx = json.load(f)  # EventId -> int index

    templates_df = pd.read_csv(templates_file)
    # Build idx → template text
    idx_to_template = {}
    for _, row in templates_df.iterrows():
        event_id = row["EventId"]
        template_text = str(row.get("EventTemplate", row.get("EventId", "")))
        idx = event_id_to_idx.get(event_id)
        if idx is not None:
            idx_to_template[idx] = template_text

    with open(template_text_file, "w") as f:
        json.dump(idx_to_template, f)
    print(f"  Saved {len(idx_to_template)} template texts")
