"""
Pairwise log datasets for SpikeLog-v2.

Training: pairs of sessions, label encodes relative anomalousness.
    - 50% normal-normal pairs → y=0.0
    - 25% normal-anomaly pairs → y=4.0
    - 25% anomaly-anomaly pairs → y=8.0

Test inference: each test sample compared against n_comparisons anomalous +
n_comparisons normal references; averaged score → threshold → binary prediction.

Each session is a list of event indices.  The __getitem__ methods look up the
pre-computed event_vectors (EventIdx → 300-dim float32 numpy array) and return
padded tensors.  Index 0 is reserved for padding (all zeros).
"""

import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PairwiseTrainDataset(Dataset):
    """Pairwise training dataset.

    Args:
        train_normal_file : path to train_normal.pkl  (list of event-idx sequences)
        train_anomaly_file: path to train_anomaly.pkl (list of event-idx sequences)
        event_vectors     : np.ndarray shape (n_events+1, emb_dim)
        max_seq_len       : truncate sequences longer than this
    """

    def __init__(
        self,
        train_normal_file: str,
        train_anomaly_file: str,
        event_vectors: np.ndarray,
        max_seq_len: int = 100,
    ):
        with open(train_normal_file, "rb") as f:
            self.normal_seqs: list[list[int]] = pickle.load(f)
        with open(train_anomaly_file, "rb") as f:
            self.anomaly_seqs: list[list[int]] = pickle.load(f)

        self.event_vectors = event_vectors  # (n_events+1, emb_dim)
        self.max_seq_len = max_seq_len
        self.emb_dim = event_vectors.shape[1]

        # Match original SpikeLog: epoch size = total training samples
        # (each sample generates one pair via __getitem__ with random pairing)
        self._len = len(self.normal_seqs) + len(self.anomaly_seqs)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        pair_type = idx % 4
        if pair_type in (0, 1):
            # 50% normal-normal
            i = np.random.randint(len(self.normal_seqs))
            j = np.random.randint(len(self.normal_seqs))
            seq1 = self.normal_seqs[i]
            seq2 = self.normal_seqs[j]
            y = 0.0
        elif pair_type == 2:
            # 25% normal-anomaly
            i = np.random.randint(len(self.normal_seqs))
            j = np.random.randint(len(self.anomaly_seqs))
            seq1 = self.normal_seqs[i]
            seq2 = self.anomaly_seqs[j]
            y = 4.0
        else:
            # 25% anomaly-anomaly
            i = np.random.randint(len(self.anomaly_seqs))
            j = np.random.randint(len(self.anomaly_seqs))
            seq1 = self.anomaly_seqs[i]
            seq2 = self.anomaly_seqs[j]
            y = 8.0

        x1 = self._seq_to_tensor(seq1)
        x2 = self._seq_to_tensor(seq2)
        return x1, x2, torch.tensor(y, dtype=torch.float32)

    def _seq_to_tensor(self, seq: list[int]) -> torch.Tensor:
        """Convert event-idx list → float tensor (seq_len, emb_dim)."""
        seq = seq[:self.max_seq_len]
        if len(seq) == 0:
            return torch.zeros(1, self.emb_dim, dtype=torch.float32)
        vecs = self.event_vectors[seq]  # (seq_len, emb_dim)
        return torch.from_numpy(vecs.copy())


class PairwiseTestDataset(Dataset):
    """Test dataset for pairwise anomaly scoring.

    Each sample is compared against n_comparisons anomalous references and
    n_comparisons normal references.  The collate_fn returns:
        x_test  : (B, seq_len, emb_dim)
        x_anom  : (B, n_comparisons, ref_len, emb_dim)
        x_norm  : (B, n_comparisons, ref_len, emb_dim)
        labels  : (B,)

    Args:
        test_file         : path to test.pkl  (list of (seq, label) tuples)
        train_normal_file : path to train_normal.pkl  (reference normal pool)
        train_anomaly_file: path to train_anomaly.pkl (reference anomaly pool)
        event_vectors     : np.ndarray shape (n_events+1, emb_dim)
        n_comparisons     : number of reference samples per class
        max_seq_len       : truncate sequences
    """

    def __init__(
        self,
        test_file: str,
        train_normal_file: str,
        train_anomaly_file: str,
        event_vectors: np.ndarray,
        n_comparisons: int = 30,
        max_seq_len: int = 100,
    ):
        with open(test_file, "rb") as f:
            test_data: list[tuple[list[int], int]] = pickle.load(f)
        with open(train_normal_file, "rb") as f:
            self.normal_pool: list[list[int]] = pickle.load(f)
        with open(train_anomaly_file, "rb") as f:
            self.anomaly_pool: list[list[int]] = pickle.load(f)

        self.test_seqs = [seq for seq, _ in test_data]
        self.test_labels = [label for _, label in test_data]

        self.event_vectors = event_vectors
        self.n_comparisons = n_comparisons
        self.max_seq_len = max_seq_len
        self.emb_dim = event_vectors.shape[1]

    def __len__(self) -> int:
        return len(self.test_labels)

    def __getitem__(self, idx: int):
        x_test = self._seq_to_tensor(self.test_seqs[idx])

        # Sample n_comparisons anomalous + n_comparisons normal references
        a_idx = np.random.randint(len(self.anomaly_pool), size=self.n_comparisons)
        u_idx = np.random.randint(len(self.normal_pool), size=self.n_comparisons)

        x_anom = [self._seq_to_tensor(self.anomaly_pool[i]) for i in a_idx]
        x_norm = [self._seq_to_tensor(self.normal_pool[i]) for i in u_idx]

        label = torch.tensor(self.test_labels[idx], dtype=torch.long)
        return x_test, x_anom, x_norm, label

    def _seq_to_tensor(self, seq: list[int]) -> torch.Tensor:
        seq = seq[:self.max_seq_len]
        if len(seq) == 0:
            return torch.zeros(1, self.emb_dim, dtype=torch.float32)
        vecs = self.event_vectors[seq]
        return torch.from_numpy(vecs.copy())


# ─── Collate functions ────────────────────────────────────────────────────────

def collate_train(batch):
    """Collate for PairwiseTrainDataset.

    Returns:
        x1: (B, max_len, emb_dim)
        x2: (B, max_len, emb_dim)
        y:  (B,)
    """
    x1_list, x2_list, y_list = zip(*batch)
    x1 = _pad_sequences(x1_list)
    x2 = _pad_sequences(x2_list)
    y = torch.stack(y_list)
    return x1, x2, y


def collate_test(batch):
    """Collate for PairwiseTestDataset.

    Returns:
        x_test : (B, max_len, emb_dim)
        x_anom : (B, n_cmp, ref_len, emb_dim)
        x_norm : (B, n_cmp, ref_len, emb_dim)
        labels : (B,)
    """
    x_tests, x_anoms, x_norms, labels = zip(*batch)

    x_test = _pad_sequences(x_tests)  # (B, T, D)
    labels = torch.stack(labels)       # (B,)

    # x_anoms is list of lists: [[ref_tensor, ...], ...]
    # shape after stacking: (B, n_cmp, ref_T, D)
    B = len(x_anoms)
    n_cmp = len(x_anoms[0])

    # Find max ref length across all batch × n_cmp
    anom_max = max(t.size(0) for lst in x_anoms for t in lst)
    norm_max = max(t.size(0) for lst in x_norms for t in lst)
    emb_dim = x_test.size(-1)

    x_anom = torch.zeros(B, n_cmp, anom_max, emb_dim)
    x_norm = torch.zeros(B, n_cmp, norm_max, emb_dim)

    for b in range(B):
        for c in range(n_cmp):
            t = x_anoms[b][c]
            x_anom[b, c, :t.size(0)] = t
            t = x_norms[b][c]
            x_norm[b, c, :t.size(0)] = t

    return x_test, x_anom, x_norm, labels


def _pad_sequences(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Pad variable-length (L, D) tensors to (B, max_L, D)."""
    max_len = max(t.size(0) for t in tensors)
    emb_dim = tensors[0].size(1)
    B = len(tensors)
    padded = torch.zeros(B, max_len, emb_dim)
    for i, t in enumerate(tensors):
        padded[i, :t.size(0)] = t
    return padded
