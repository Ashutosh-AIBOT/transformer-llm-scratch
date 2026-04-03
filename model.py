"""
========================================================================
FULL TRANSFORMER MODEL — Built from Scratch (Paper-Accurate)
========================================================================
Reference: "Attention is All You Need" (Vaswani et al., 2017)

This file assembles the complete Transformer architecture:

  ┌──────────────────────────────────────┐
  │            ENCODER STACK             │
  │  ┌────────────────────────────────┐  │
  │  │  Multi-Head Self-Attention     │  │
  │  │  + Add & LayerNorm             │  │
  │  ├────────────────────────────────┤  │
  │  │  Feed-Forward Network          │  │
  │  │  + Add & LayerNorm             │  │
  │  └────────────────────────────────┘  │
  │           × N layers                 │
  └──────────────────────────────────────┘
                    ↓
            [Classification Head]
                    ↓
              (num_classes)

Shape Flow for Classification:
  Token IDs:       (batch, seq_len)
  → Embedding:     (batch, seq_len, d_model)
  → + PosEnc:      (batch, seq_len, d_model)
  → Encoder ×N:    (batch, seq_len, d_model)
  → Mean Pool:     (batch, d_model)
  → Linear Head:   (batch, num_classes)

========================================================================
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notebooks.layers import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionalEncoding,
    TokenEmbedding,
)


# ======================================================================
# ENCODER LAYER (Single Block)
# ======================================================================
# Each encoder layer has two sub-layers:
#   1. Multi-Head Self-Attention
#   2. Position-Wise Feed-Forward
#
# Each sub-layer uses a residual connection followed by layer norm:
#   output = LayerNorm(x + SubLayer(x))
#
# This "Add & Norm" pattern prevents vanishing gradients and allows
# the model to scale to many layers (the paper uses N=6).
# ======================================================================

class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder block.

    Shape flow:
        Input:  (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)        ← same shape (residual!)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: Self-Attention + Add & Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: Feed-Forward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# ======================================================================
# TRANSFORMER ENCODER (Full Stack)
# ======================================================================
# Stacks N EncoderLayers on top of each other.
# Includes Token Embedding + Positional Encoding at the input.
# Tracks intermediate tensor shapes for dashboard visualization.
# ======================================================================

class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder with shape tracking.

    Shape flow:
        Input:  (batch, seq_len)                — token IDs
        Output: (batch, seq_len, d_model)       — contextual representations
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 max_seq_len=128, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Input processing
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Stack of N encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

        # ── Shape Tracker (for visualization) ─────────────────────────
        self.shape_log = {}

    def forward(self, x, mask=None):
        """
        Forward pass with shape tracking at every stage.
        """
        self.shape_log = {}

        # Stage 0: Raw token IDs
        self.shape_log["0_Input_Tokens"] = list(x.shape)

        # Stage 1: Token Embedding
        x = self.token_emb(x)
        self.shape_log["1_Token_Embedding"] = list(x.shape)

        # Stage 2: Add Positional Encoding
        x = self.pos_enc(x)
        self.shape_log["2_Positional_Encoding"] = list(x.shape)

        # Stage 3-N: Encoder Layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            self.shape_log[f"3_Encoder_Layer_{i+1}"] = list(x.shape)

        # Final normalization
        x = self.norm(x)
        self.shape_log["4_Final_Norm"] = list(x.shape)

        return x


# ======================================================================
# TRANSFORMER CLASSIFIER
# ======================================================================
# Adds a classification head on top of the Encoder.
# Uses mean pooling across the sequence dimension to get a
# fixed-size representation, then projects to num_classes.
# ======================================================================

class TransformerClassifier(nn.Module):
    """
    Transformer Encoder + Classification Head.

    Used for: StackOverflow programming language classification.

    Shape flow:
        (batch, seq_len) → Encoder → (batch, seq_len, d_model)
                         → Mean Pool → (batch, d_model)
                         → Linear → (batch, num_classes)
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff,
                 num_classes, max_seq_len=128, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, n_layers, n_heads, d_ff,
            max_seq_len, dropout
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # Encode: (batch, seq_len) → (batch, seq_len, d_model)
        enc_out = self.encoder(x, mask)

        # Mean Pool: (batch, seq_len, d_model) → (batch, d_model)
        pooled = enc_out.mean(dim=1)

        # Classify: (batch, d_model) → (batch, num_classes)
        logits = self.classifier(pooled)

        return logits


# ======================================================================
# SELF-TEST
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FULL TRANSFORMER MODEL — End-to-End Shape Test")
    print("=" * 60)

    # Config matching our GTX 1650 optimized setup
    VOCAB    = 5000
    D_MODEL  = 256
    N_LAYERS = 3
    N_HEADS  = 8
    D_FF     = 1024
    N_CLASS  = 20
    SEQ_LEN  = 128

    model = TransformerClassifier(
        VOCAB, D_MODEL, N_LAYERS, N_HEADS, D_FF, N_CLASS, SEQ_LEN
    )

    # Simulate a batch of 4 sequences
    dummy = torch.randint(0, VOCAB, (4, SEQ_LEN))
    output = model(dummy)

    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {output.shape}")

    print("\n── Shape Flow (Every Stage) ──")
    for stage, shape in model.encoder.shape_log.items():
        print(f"  [{stage}]  →  {shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")
    print(f"📊 Estimated VRAM: ~{total_params * 4 / 1024**2:.0f} MB (fp32)")

    print("\n✅ FULL TRANSFORMER TEST PASSED")
