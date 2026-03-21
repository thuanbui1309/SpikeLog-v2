"""
Semantic embedding generation for SpikeLog-v2.

Matches SpikeLog (TKDE 2024) embedding pipeline exactly:
- Pretrained FastText word vectors (300d) — NOT self-trained
- Tokenize templates on delimiters (TF-IDF at token level)
- Camel-case split only for sub-token lookup in pretrained vectors
- TF-IDF weighted average per template

Pipeline:
    templates_text.json (EventIdx → template string)
    → tokenize (split on delimiters, keep original case)
    → compute IDF across templates (at token level)
    → for each token: camel-case split → lookup pretrained → average sub-tokens
    → TF-IDF weighted sum per template
    → save event_vectors.npy
"""

import os
import re
import json
import math
import logging
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

# Default pretrained model for gensim.downloader
_DEFAULT_PRETRAINED = "fasttext-wiki-news-subwords-300"


def generate_event_vectors(config: dict, project_root: str):
    """Main entry point: generate 300-dim semantic vectors for each event template.

    Reads templates_text.json (EventIdx → template string) from the preprocessed
    output directory and saves event_vectors.npy.

    Returns:
        str: path to event_vectors.npy
    """
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    dataset = ds_cfg["name"]
    embedding_dim = data_cfg.get("embedding_dim", 300)

    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    template_text_file = os.path.join(output_dir, "templates_text.json")
    vectors_file = os.path.join(output_dir, "event_vectors.npy")

    if os.path.exists(vectors_file):
        print(f"[Embed] Event vectors already exist at {vectors_file}")
        return vectors_file

    if not os.path.exists(template_text_file):
        raise FileNotFoundError(
            f"templates_text.json not found at {template_text_file}. "
            "Run preprocessing first."
        )

    with open(template_text_file, "r") as f:
        idx_to_template = json.load(f)  # str(int) → template string

    # Convert keys to int
    idx_to_template = {int(k): v for k, v in idx_to_template.items()}

    print(f"[Embed] Generating embeddings for {len(idx_to_template)} templates "
          f"(dim={embedding_dim})...")

    # Tokenize all templates (pre-camel-case tokens, for TF-IDF)
    idx_to_tokens = {idx: _tokenize_template(tmpl)
                     for idx, tmpl in idx_to_template.items()}

    # Load pretrained word vectors
    word_vectors = _load_pretrained_vectors(data_cfg, project_root)

    # Compute IDF at token level (before camel-case split, matching SpikeLog)
    idf = _compute_idf(idx_to_tokens)

    # Compute TF-IDF weighted average per template
    max_idx = max(idx_to_tokens.keys())
    # Row 0 = padding (all zeros), rows 1..max_idx = event vectors
    vectors = np.zeros((max_idx + 1, embedding_dim), dtype=np.float32)

    n_oov_total = 0
    n_subtoken_total = 0

    for idx, tokens in idx_to_tokens.items():
        vec, n_oov, n_sub = _tfidf_weighted_average(
            tokens, word_vectors, idf, embedding_dim
        )
        vectors[idx] = vec
        n_oov_total += n_oov
        n_subtoken_total += n_sub

    np.save(vectors_file, vectors)
    coverage = 100 * (1 - n_oov_total / max(n_subtoken_total, 1))
    print(f"[Embed] Saved event_vectors.npy: shape={vectors.shape}")
    print(f"[Embed] Sub-token coverage: "
          f"{n_subtoken_total - n_oov_total}/{n_subtoken_total} ({coverage:.1f}%)")

    return vectors_file


def load_event_vectors(config: dict, project_root: str) -> np.ndarray:
    """Load pre-generated event vectors. Returns shape (n_events+1, embedding_dim)."""
    ds_cfg = config["dataset"]
    data_cfg = config["data"]
    dataset = ds_cfg["name"]
    output_dir = os.path.join(project_root, data_cfg["output_dir"], dataset)
    vectors_file = os.path.join(output_dir, "event_vectors.npy")
    if not os.path.exists(vectors_file):
        raise FileNotFoundError(
            f"event_vectors.npy not found. Run embedding generation first."
        )
    return np.load(vectors_file)


# ─── Pretrained vector loading ───────────────────────────────────────────────

def _load_pretrained_vectors(data_cfg, project_root):
    """Load pretrained word vectors.

    Checks for a local .vec file first (data.pretrained_vectors config).
    Falls back to auto-download via gensim.downloader.
    """
    pretrained_path = data_cfg.get("pretrained_vectors")

    if pretrained_path:
        full_path = os.path.join(project_root, pretrained_path)
        if os.path.exists(full_path):
            from gensim.models import KeyedVectors
            print(f"[Embed] Loading pretrained vectors from {full_path}...")
            return KeyedVectors.load_word2vec_format(full_path, binary=False)
        else:
            print(f"[Embed] Warning: pretrained_vectors path not found: {full_path}")

    # Auto-download via gensim.downloader
    print(f"[Embed] Loading pretrained FastText vectors ({_DEFAULT_PRETRAINED})...")
    print(f"[Embed] First run will download ~600MB. Cached in ~/gensim-data/")
    import gensim.downloader as api
    return api.load(_DEFAULT_PRETRAINED)


# ─── Tokenization ────────────────────────────────────────────────────────────

def _tokenize_template(template: str) -> list[str]:
    """Tokenize a log template string into tokens (original case, no camel-case split).

    Matches SpikeLog's tokenization: split on delimiters, remove standalone - and _.
    Camel-case splitting is applied later during embedding lookup only.
    """
    # Replace Drain wildcard <*> with empty
    template = template.replace("<*>", " ")

    # Split on delimiters (same regex as SpikeLog/PLELog)
    raw_tokens = re.split(r'[,\!:=\[\]\(\)\$\s\.\/\#\|\\]', template.strip())

    # Remove standalone - and _ tokens
    cleaned = []
    for tok in raw_tokens:
        if re.match(r'^[-_]+$', tok):
            continue
        if tok.strip():
            cleaned.append(tok)

    return cleaned


def _camel_to_tokens(token: str) -> list[str]:
    """Split CamelCase/snake_case/digit boundaries into lowercase sub-tokens.

    Adapted from PLELog/data/Embedding.py: like_camel_to_tokens().
    Used for looking up sub-tokens in pretrained word vectors.
    """
    simple_format = []
    temp = ''
    flag = False  # previous char was uppercase

    for i, ch in enumerate(token):
        if ch in ('-', '_'):
            simple_format.append(temp)
            temp = ''
            flag = False
        elif ch.isdigit():
            simple_format.append(temp)
            simple_format.append(ch)
            temp = ''
            flag = False
        elif ch.islower():
            if flag and len(temp) > 1:
                # e.g. "ABc" → ["A", "Bc"] — last uppercase was start of new word
                w = temp[-1]
                temp = temp[:-1]
                simple_format.append(temp)
                temp = w + ch
            else:
                temp += ch
            flag = False
        else:  # uppercase
            if not flag:
                simple_format.append(temp)
                temp = ''
            temp += ch.lower()
            flag = True

    simple_format.append(temp)
    return [t for t in simple_format if t.strip()]


def _token_to_embedding(token: str, word_vectors, embedding_dim: int):
    """Get embedding for a token by camel-case splitting and averaging sub-token lookups.

    Matches SpikeLog: for each token, camel-case split into sub-tokens,
    look up each in pretrained vectors, average all sub-token embeddings.
    OOV sub-tokens contribute zero vector (same as SpikeLog).

    Returns:
        (embedding, n_oov, n_subtokens)
    """
    sub_tokens = _camel_to_tokens(token)
    if not sub_tokens:
        return np.zeros(embedding_dim, dtype=np.float64), 0, 0

    emb = np.zeros(embedding_dim, dtype=np.float64)
    n_oov = 0
    for st in sub_tokens:
        if st in word_vectors:
            emb += word_vectors[st]
        else:
            n_oov += 1
            # OOV → zero vector (matching SpikeLog)

    emb = emb / len(sub_tokens)
    return emb, n_oov, len(sub_tokens)


# ─── IDF + TF-IDF weighting ──────────────────────────────────────────────────

def _compute_idf(idx_to_tokens: dict) -> dict:
    """Compute IDF scores at token level (before camel-case split).

    IDF(word) = log(N / df), where N = number of templates,
    df = number of templates containing the word.
    Matches SpikeLog's IDF computation.
    """
    total = len(idx_to_tokens)
    doc_freq = Counter()
    for tokens in idx_to_tokens.values():
        for word in set(tokens):
            doc_freq[word] += 1

    idf = {}
    for word, df in doc_freq.items():
        idf[word] = math.log(total / df) if df > 0 else 1.0
    return idf


def _tfidf_weighted_average(
    tokens: list[str],
    word_vectors,
    idf: dict,
    embedding_dim: int,
):
    """Compute TF-IDF weighted average of token embeddings for a template.

    Matches SpikeLog's computation:
    - TF and IDF at token level (before camel-case split)
    - Token embedding = average of camel-case sub-token lookups in pretrained vectors
    - Template embedding = sum(TF * IDF * token_embedding)

    Returns:
        (embedding, n_oov_total, n_subtokens_total)
    """
    if len(tokens) == 0:
        return np.zeros(embedding_dim, dtype=np.float32), 0, 0

    vec = np.zeros(embedding_dim, dtype=np.float64)
    token_counts = Counter(tokens)
    n = len(tokens)
    n_oov_total = 0
    n_sub_total = 0

    for token, count in token_counts.items():
        tf = count / n
        idf_score = idf.get(token, 1.0)

        # Get token embedding via camel-case sub-token lookup
        token_emb, n_oov, n_sub = _token_to_embedding(
            token, word_vectors, embedding_dim
        )
        vec += tf * idf_score * token_emb
        n_oov_total += n_oov
        n_sub_total += n_sub

    return vec.astype(np.float32), n_oov_total, n_sub_total
