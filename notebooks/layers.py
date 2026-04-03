"""
========================================================================
TRANSFORMER LAYERS — Built Entirely from Scratch
========================================================================
Reference: "Attention is All You Need" (Vaswani et al., 2017)
Paper URL: https://arxiv.org/abs/1706.03762

This file implements every sub-layer of the Transformer architecture
using only basic PyTorch operations (no nn.Transformer shortcuts).

Components:
  1. ScaledDotProductAttention  — The heart of every Transformer
  2. MultiHeadAttention         — Parallel attention in h subspaces
  3. PositionWiseFeedForward    — Two-layer MLP per token position
  4. PositionalEncoding         — Sinusoidal "clock" for word order
  5. TokenEmbedding             — Lookup table + sqrt(d_model) scaling
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# 1. SCALED DOT-PRODUCT ATTENTION
# ======================================================================
# Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#
# Why scale by sqrt(d_k)?
#   When d_k is large, the dot products QK^T grow large in magnitude,
#   pushing softmax into regions with extremely small gradients.
#   Dividing by sqrt(d_k) counteracts this effect.
# ======================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention.

    Input shapes:
        q: (batch, n_heads, seq_len, d_k)
        k: (batch, n_heads, seq_len, d_k)
        v: (batch, n_heads, seq_len, d_k)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Output shapes:
        output: (batch, n_heads, seq_len, d_k)
        attn_weights: (batch, n_heads, seq_len, seq_len)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        # Step 1: Compute raw attention scores  →  (batch, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: Apply mask (padding mask or causal mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Softmax to get attention probabilities
        attn_weights = F.softmax(scores, dim=-1)

        # Step 4: Weighted sum of values
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


# ======================================================================
# 2. MULTI-HEAD ATTENTION
# ======================================================================
# Instead of performing a single attention function, we linearly
# project Q, K, V into h different subspaces (heads), perform
# attention in parallel, concatenate, and project back.
#
# MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W^O
# where head_i = Attention(Q*W_i^Q, K*W_i^K, V*W_i^V)
# ======================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Total model dimension (e.g., 256)
        n_heads: Number of parallel attention heads (e.g., 8)
        dropout: Dropout rate for attention weights

    Shape flow:
        Input:  (batch, seq_len, d_model)   e.g., (32, 128, 256)
        Split:  (batch, n_heads, seq_len, d_k)  e.g., (32, 8, 128, 32)
        Output: (batch, seq_len, d_model)   e.g., (32, 128, 256)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimension per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection (W^O in the paper)
        self.fc_out = nn.Linear(d_model, d_model)

        # Core attention module
        self.attention = ScaledDotProductAttention()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attn_weights = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Step 1: Project Q, K, V and reshape into multiple heads
        # (batch, seq, d_model) → (batch, seq, n_heads, d_k) → (batch, n_heads, seq, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Step 2: Compute scaled dot-product attention for all heads
        x, self.attn_weights = self.attention(q, k, v, mask)

        # Step 3: Concatenate heads back together
        # (batch, n_heads, seq, d_k) → (batch, seq, n_heads, d_k) → (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Step 4: Final linear projection (W^O)
        return self.fc_out(x)


# ======================================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ======================================================================
# FFN(x) = max(0, x*W1 + b1) * W2 + b2
#
# This is applied to each position separately and identically.
# The inner layer has d_ff neurons (typically 4 * d_model).
# It gives the model "thinking capacity" after attention.
# ======================================================================

class PositionWiseFeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each token position.

    Shape flow:
        Input:  (batch, seq_len, d_model)   e.g., (32, 128, 256)
        Hidden: (batch, seq_len, d_ff)      e.g., (32, 128, 1024)
        Output: (batch, seq_len, d_model)   e.g., (32, 128, 256)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand → ReLU → Contract
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ======================================================================
# 4. POSITIONAL ENCODING (Sinusoidal)
# ======================================================================
# Since Transformers have no recurrence (no RNN), they have no
# inherent sense of word order. "Man bites dog" == "Dog bites man".
#
# We inject position information using sine and cosine waves:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# Why sinusoidal?
#   1. Each dimension oscillates at a different frequency.
#   2. The model can learn to attend to relative positions because
#      PE(pos+k) can be represented as a linear function of PE(pos).
# ======================================================================

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encodings into token embeddings.

    Shape flow:
        Input:  (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)   [values shifted by position signals]
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # Odd indices

        # Shape: (1, max_len, d_model) — the 1 is for batch broadcasting
        pe = pe.unsqueeze(0)

        # register_buffer: saved with model, but NOT trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to embeddings, then apply dropout
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ======================================================================
# 5. TOKEN EMBEDDING (with sqrt(d_model) scaling)
# ======================================================================
# The paper scales embeddings by sqrt(d_model) to keep them
# proportional to the positional encoding magnitudes.
# ======================================================================

class TokenEmbedding(nn.Module):
    """
    Converts token IDs to dense vectors and scales by sqrt(d_model).

    Shape flow:
        Input:  (batch, seq_len)         — integer token IDs
        Output: (batch, seq_len, d_model) — dense floating-point vectors
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ======================================================================
# SELF-TEST
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFORMER LAYERS — Shape Verification Test")
    print("=" * 60)

    B, S, D, H = 2, 10, 256, 8
    print(f"\nConfig: batch={B}, seq_len={S}, d_model={D}, n_heads={H}")

    # Test MHA
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, S, D)
    out = mha(x, x, x)
    print(f"\n[MHA] Input: {x.shape} → Output: {out.shape}")
    assert out.shape == x.shape, "MHA shape mismatch!"

    # Test FFN
    ffn = PositionWiseFeedForward(D, D * 4)
    out_ffn = ffn(x)
    print(f"[FFN] Input: {x.shape} → Output: {out_ffn.shape}")
    assert out_ffn.shape == x.shape, "FFN shape mismatch!"

    # Test Positional Encoding
    pe = PositionalEncoding(D)
    out_pe = pe(x)
    print(f"[PE]  Input: {x.shape} → Output: {out_pe.shape}")

    # Test Token Embedding
    emb = TokenEmbedding(1000, D)
    tokens = torch.randint(0, 1000, (B, S))
    out_emb = emb(tokens)
    print(f"[EMB] Input: {tokens.shape} → Output: {out_emb.shape}")

    print("\n✅ ALL LAYER TESTS PASSED")
