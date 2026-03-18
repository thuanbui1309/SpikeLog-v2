"""
Semantic embedding generation for SpikeLog-v2.

Implements Word2Vec + TF-IDF weighted template embeddings, following SpikeLog (TKDE 2024).
Key differences from SpikeLog reference:
- Trains Word2Vec on log template tokens (self-contained, no external fastText file)
- Returns EventIdx (int) → 300-dim numpy array (instead of template string → array)
- Camel-case splitting for token normalization (from PLELog/data/Embedding.py)

Pipeline:
    templates_text.json (EventIdx → template string)
    → tokenize (split + camel-case)
    → train Word2Vec on all tokens
    → compute IDF across templates
    → TF-IDF weighted average per template
    → save event_vectors.npy + vocab.json
"""

import os
import re
import json
import math
import logging
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)


def generate_event_vectors(config: dict, project_root: str):
    """Main entry point: generate 300-dim semantic vectors for each event template.

    Reads templates_text.json (EventIdx → template string) from the preprocessed
    output directory and saves:
    - event_vectors.npy  : shape (max_event_idx + 1, embedding_dim), row i = event i
    - vocab.json         : word → int index (for inspection)

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

    # Tokenize all templates
    idx_to_tokens = {idx: _tokenize_template(tmpl)
                     for idx, tmpl in idx_to_template.items()}

    # Train Word2Vec on all tokens
    all_sentences = list(idx_to_tokens.values())
    word_vectors = _train_word2vec(all_sentences, embedding_dim)

    # Compute IDF over templates
    idf = _compute_idf(idx_to_tokens)

    # Compute TF-IDF weighted average per template
    max_idx = max(idx_to_tokens.keys())
    # Row 0 = padding (all zeros), rows 1..max_idx = event vectors
    vectors = np.zeros((max_idx + 1, embedding_dim), dtype=np.float32)

    for idx, tokens in idx_to_tokens.items():
        vec = _tfidf_weighted_average(tokens, word_vectors, idf, embedding_dim)
        vectors[idx] = vec

    np.save(vectors_file, vectors)
    print(f"[Embed] Saved event_vectors.npy: shape={vectors.shape}")

    # Save vocab for inspection
    vocab = {word: i for i, word in enumerate(word_vectors.keys())}
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

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


# ─── Tokenization ────────────────────────────────────────────────────────────

def _tokenize_template(template: str) -> list[str]:
    """Tokenize a log template string into lowercase tokens.

    Steps:
    1. Split on common delimiters: space, comma, =, :, [, ], (, ), $, ., /, #, |, \\
    2. Apply camel-case splitting on each token (from PLELog)
    3. Filter empty tokens and pure wildcards
    """
    # Replace Drain wildcard <*> with empty
    template = template.replace("<*>", " ")

    # Split on delimiters
    raw_tokens = re.split(r'[,\!:=\[\]\(\)\$\s\.\/\#\|\\]', template.strip())

    # Remove - and _ only tokens
    cleaned = []
    for tok in raw_tokens:
        if re.match(r'^[-_]+$', tok):
            continue
        if tok.strip():
            cleaned.append(tok)

    # Camel-case split each token
    result = []
    for tok in cleaned:
        subtokens = _camel_to_tokens(tok)
        result.extend(subtokens)

    return [t.lower() for t in result if t.strip()]


def _camel_to_tokens(token: str) -> list[str]:
    """Split CamelCase/snake_case/digit boundaries into sub-tokens.

    Adapted from PLELog/data/Embedding.py: like_camel_to_tokens().
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


# ─── Word2Vec (via gensim) ────────────────────────────────────────────────────

def _train_word2vec(sentences: list[list[str]], embedding_dim: int = 300) -> dict:
    """Train Word2Vec on tokenized templates. Returns word → numpy array."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim not found. Install: pip install gensim")

    # Filter empty sentences
    sentences = [s for s in sentences if len(s) > 0]

    if len(sentences) == 0:
        print("  Warning: no tokens found, using random vectors")
        return {}

    # Use skip-gram (sg=1) for better quality on small corpora
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,   # keep all tokens (log vocab is small)
        workers=4,
        sg=1,          # skip-gram
        epochs=10,
        seed=42,
    )

    word_vectors = {word: model.wv[word] for word in model.wv.key_to_index}
    print(f"  Word2Vec trained on {len(sentences)} templates, vocab={len(word_vectors)}")
    return word_vectors


# ─── IDF + TF-IDF weighting ──────────────────────────────────────────────────

def _compute_idf(idx_to_tokens: dict) -> dict:
    """Compute IDF scores: word → log(N / df).

    N = number of templates, df = number of templates containing the word.
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
    word_vectors: dict,
    idf: dict,
    embedding_dim: int,
) -> np.ndarray:
    """Compute TF-IDF weighted average of token embeddings for a template.

    TF = count(token) / len(tokens), IDF from corpus-level IDF dict.
    OOV tokens contribute zero vector.
    """
    if len(tokens) == 0:
        return np.zeros(embedding_dim, dtype=np.float32)

    vec = np.zeros(embedding_dim, dtype=np.float64)
    token_counts = Counter(tokens)
    n = len(tokens)

    for token, count in token_counts.items():
        if token not in word_vectors:
            continue
        tf = count / n
        idf_score = idf.get(token, 1.0)
        vec += tf * idf_score * word_vectors[token]

    return vec.astype(np.float32)
