"""
========================================================================
STOP 01 — DATA ENGINEERING & FEATURE ENGINEERING
========================================================================
StackOverflow Programming Questions (2020-2025)
Task: Classify the programming language of a question from its text.

Pipeline:
  1. Load raw CSV (95,636 rows)
  2. Clean HTML tags, normalize whitespace
  3. Combine title + body into a single text field
  4. Build a character-level tokenizer (fast, low-memory, no OOV)
  5. Encode labels (programming_language → integer)
  6. Save processed tensors to disk

Why character-level tokenizer?
  - No out-of-vocabulary (OOV) tokens
  - Works perfectly for code snippets (e.g., "pd.DataFrame()")
  - Extremely fast to build (no BPE training needed)
  - Compact vocabulary (~100 chars) fits easily on GTX 1650
========================================================================
"""

import pandas as pd
import numpy as np
import torch
import re
import json
import sys
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_RAW, DATA_PROCESSED, MODELS


# ── Step 1: HTML Cleaning ─────────────────────────────────────────────
def clean_text(text):
    """Remove HTML tags, URLs, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]*>', ' ', text)          # Strip HTML tags
    text = re.sub(r'http\S+', '', text)            # Strip URLs
    text = re.sub(r'\s+', ' ', text).strip()       # Normalize spaces
    return text.lower()


# ── Step 2: Character-Level Tokenizer ─────────────────────────────────
class CharTokenizer:
    """
    Maps every character to an integer ID.

    Special tokens:
      0 = [PAD]   — padding for shorter sequences
      1 = [UNK]   — unknown characters

    Why characters instead of words?
      Code contains symbols like (){}[]= that word tokenizers struggle with.
      Character-level handles EVERYTHING.
    """
    def __init__(self):
        self.char2idx = {'[PAD]': 0, '[UNK]': 1}
        self.idx2char = {0: '[PAD]', 1: '[UNK]'}

    def build(self, texts, min_freq=5):
        """Build vocabulary from a list of texts."""
        print("  [Tokenizer] Counting character frequencies...")
        counter = Counter()
        for text in tqdm(texts, desc="  Building vocab"):
            counter.update(text)

        # Add characters that appear at least min_freq times
        idx = 2
        for char, count in counter.most_common():
            if count >= min_freq:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        print(f"  [Tokenizer] Vocabulary size: {len(self.char2idx)}")
        return self

    def encode(self, text, max_len=128):
        """Convert text to a fixed-length tensor of character IDs."""
        ids = [self.char2idx.get(c, 1) for c in text[:max_len]]
        # Pad to max_len
        ids += [0] * (max_len - len(ids))
        return ids

    @property
    def vocab_size(self):
        return len(self.char2idx)

    def save(self, path):
        """Save tokenizer to JSON."""
        with open(path, 'w') as f:
            json.dump(self.char2idx, f)

    def load(self, path):
        """Load tokenizer from JSON."""
        with open(path, 'r') as f:
            self.char2idx = json.load(f)
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        return self


# ── Step 3: Full Pipeline ─────────────────────────────────────────────
def run_data_pipeline(max_seq_len=128, test_size=0.2):
    """
    Complete data engineering pipeline.

    Returns: Saves train/val tensors + tokenizer + label map to disk.
    """
    print("=" * 60)
    print("STOP 01 — DATA ENGINEERING PIPELINE")
    print("=" * 60)

    csv_path = DATA_RAW / "stackoverflow_combined.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # ── Load ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading raw CSV...")
    df = pd.read_csv(csv_path)
    print(f"       Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Clean ─────────────────────────────────────────────────────────
    print("\n[2/6] Cleaning HTML & combining title + body...")
    df['text'] = df['title'].fillna('') + ' ' + df['body'].apply(clean_text)
    df['text'] = df['text'].apply(lambda x: x[:500])   # Cap at 500 chars for speed

    # Drop rows with empty text
    df = df[df['text'].str.len() > 10].copy()
    print(f"       Cleaned dataset: {len(df):,} rows")

    # ── Label Encoding ────────────────────────────────────────────────
    print("\n[3/6] Encoding programming language labels...")
    label2idx = {lang: i for i, lang in enumerate(sorted(df['programming_language'].unique()))}
    idx2label = {i: lang for lang, i in label2idx.items()}

    df['label'] = df['programming_language'].map(label2idx)
    num_classes = len(label2idx)
    print(f"       Classes: {num_classes}")
    for lang, idx in sorted(label2idx.items(), key=lambda x: x[1]):
        count = (df['label'] == idx).sum()
        print(f"         [{idx:2d}] {lang:15s} → {count:,} samples")

    # ── Tokenize ──────────────────────────────────────────────────────
    print("\n[4/6] Building character-level tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build(df['text'].tolist())

    print(f"\n[5/6] Encoding all sequences (max_len={max_seq_len})...")
    encoded = []
    for text in tqdm(df['text'].tolist(), desc="  Encoding"):
        encoded.append(tokenizer.encode(text, max_len=max_seq_len))

    X = torch.tensor(encoded, dtype=torch.long)
    y = torch.tensor(df['label'].values, dtype=torch.long)

    # ── Train/Val Split ───────────────────────────────────────────────
    print(f"\n[6/6] Splitting train/val ({1-test_size:.0%}/{test_size:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"       Train: {X_train.shape[0]:,} samples")
    print(f"       Val:   {X_val.shape[0]:,} samples")

    # ── Save Everything ───────────────────────────────────────────────
    torch.save(X_train, DATA_PROCESSED / "X_train.pt")
    torch.save(X_val,   DATA_PROCESSED / "X_val.pt")
    torch.save(y_train, DATA_PROCESSED / "y_train.pt")
    torch.save(y_val,   DATA_PROCESSED / "y_val.pt")

    tokenizer.save(MODELS / "tokenizer.json")

    with open(MODELS / "label_map.json", 'w') as f:
        json.dump({'label2idx': label2idx, 'idx2label': {str(k): v for k, v in idx2label.items()}}, f)

    config = {
        'vocab_size': tokenizer.vocab_size,
        'num_classes': num_classes,
        'max_seq_len': max_seq_len,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 1024,
        'dropout': 0.1,
    }
    with open(MODELS / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ STOP 01 COMPLETE — All artifacts saved to:")
    print(f"   Data:      {DATA_PROCESSED}")
    print(f"   Tokenizer: {MODELS / 'tokenizer.json'}")
    print(f"   Labels:    {MODELS / 'label_map.json'}")
    print(f"   Config:    {MODELS / 'config.json'}")
    print("=" * 60)


if __name__ == "__main__":
    run_data_pipeline()
