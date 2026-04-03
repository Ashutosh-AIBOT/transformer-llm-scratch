"""
========================================================================
VISUALIZATION ENGINE — Transformer Internals
========================================================================
All visualizations are built with pure Plotly (no JS/CSS needed).

Functions:
  1. plot_attention_heatmap     — Which tokens attend to which
  2. plot_shape_flow            — Tensor shape at every stage
  3. plot_positional_encoding   — Sinusoidal position signals
  4. plot_architecture_diagram  — Vertical encoder flow chart
  5. plot_training_history      — Loss & accuracy curves
========================================================================
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import math


# ── 1. ATTENTION HEATMAP ─────────────────────────────────────────────
def plot_attention_heatmap(attn_weights, head_idx=0, tokens=None):
    """
    Visualize attention weights for a specific head.

    Args:
        attn_weights: (n_heads, seq_len, seq_len) tensor
        head_idx: which head to visualize
        tokens: optional list of token labels for axes
    """
    matrix = attn_weights[head_idx].detach().cpu().numpy()

    labels = tokens if tokens else [f"t{i}" for i in range(matrix.shape[0])]

    fig = px.imshow(
        matrix,
        x=labels, y=labels,
        labels=dict(x="Key (Attended To)", y="Query (Looking From)", color="Weight"),
        color_continuous_scale="Viridis",
        title=f"Attention Weights — Head {head_idx}"
    )
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=450,
    )
    return fig


# ── 2. SHAPE FLOW TABLE ──────────────────────────────────────────────
def make_shape_flow_df(shape_log):
    """
    Convert the model's shape_log dict into a styled DataFrame.
    """
    rows = []
    for stage, shape in shape_log.items():
        label = stage.replace("_", " ").title()
        rows.append({
            "Stage": label,
            "Tensor Shape": str(shape),
            "Dimensions": len(shape),
            "Total Elements": np.prod(shape),
        })
    return pd.DataFrame(rows)


# ── 3. POSITIONAL ENCODING HEATMAP ───────────────────────────────────
def plot_positional_encoding(d_model=256, max_len=100):
    """
    Visualize the sinusoidal positional encoding matrix.
    Shows how each position gets a unique "fingerprint" of sine/cosine values.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    fig = px.imshow(
        pe.numpy(),
        labels=dict(x="Embedding Dimension", y="Sequence Position", color="Value"),
        color_continuous_scale="RdBu",
        title="Positional Encoding Heatmap (Sinusoidal)"
    )
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
        height=400,
    )
    return fig


# ── 4. ARCHITECTURE DIAGRAM ──────────────────────────────────────────
def plot_architecture_diagram(n_layers=3):
    """
    Vertical flow chart of the Transformer Encoder using Plotly scatter.
    No JavaScript required — pure Python.
    """
    # Build node list
    nodes = ["Input Token IDs", "Token Embedding\n× √d_model", "Positional Encoding\n(Sine/Cosine)"]
    for i in range(1, n_layers + 1):
        nodes.append(f"Encoder Layer {i}\n[MHA → Add&Norm → FFN → Add&Norm]")
    nodes.extend(["Final LayerNorm", "Mean Pooling\n(seq → vector)", "Classification Head\n(Linear → Softmax)"])

    x = [1] * len(nodes)
    y = list(range(len(nodes) - 1, -1, -1))

    colors = (
        ["#58a6ff"] * 3 +                      # input processing (blue)
        ["#238636"] * n_layers +                # encoder layers (green)
        ["#d29922", "#d29922", "#f85149"]        # output processing (gold, red)
    )

    fig = go.Figure()

    # Nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        marker=dict(size=50, color=colors, line=dict(width=2, color="#30363d")),
        text=nodes,
        textposition="middle right",
        textfont=dict(size=11, color="#e6edf3"),
        hoverinfo="text",
    ))

    # Arrows
    for i in range(len(y) - 1):
        fig.add_annotation(
            x=x[i+1], y=y[i+1] + 0.1, ax=x[i], ay=y[i] - 0.1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1. , arrowwidth=2,
            arrowcolor="#8b949e",
        )

    fig.update_layout(
        title="Transformer Encoder — Layer-by-Layer Flow",
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.4, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=120 * len(nodes),
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── 5. TRAINING HISTORY CURVES ───────────────────────────────────────
def plot_training_history(history):
    """
    Dual-axis plot: Loss curves + Accuracy curve.
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_loss'],
        mode='lines+markers', name='Train Loss',
        line=dict(color='#58a6ff', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'],
        mode='lines+markers', name='Val Loss',
        line=dict(color='#f85149', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_acc'],
        mode='lines+markers', name='Val Accuracy',
        line=dict(color='#238636', width=2, dash='dot'),
        yaxis='y2',
    ))

    fig.update_layout(
        title="Training History",
        template="plotly_dark",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Accuracy", side="right", overlaying="y",
                    showgrid=False, range=[0, 1]),
        legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center"),
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    return fig
