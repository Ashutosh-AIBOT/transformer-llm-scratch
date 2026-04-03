"""
========================================================================
TRANSFORMER FROM SCRATCH — LUXURY DASHBOARD
========================================================================
A premium Streamlit interface for exploring every internal detail
of the Transformer architecture.

Tabs:
  1. 🧠 Live Classification   — Predict programming language + see shapes
  2. 🏗️ Architecture Blueprint — Visual encoder flow + paper equations
  3. 📊 Training Analytics     — Loss curves & accuracy metrics
  4. 🔬 Attention Lab          — Multi-head attention heatmaps
  5. 📐 Paper Reference        — Full "Attention is All You Need" breakdown
========================================================================
"""

import streamlit as st
import torch
import json
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model import TransformerClassifier
from notebooks.s01_stack_data import CharTokenizer
from viz_engine import (
    plot_attention_heatmap,
    make_shape_flow_df,
    plot_positional_encoding,
    plot_architecture_diagram,
    plot_training_history,
)
from path_utils import MODELS


# ── Streamlit Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="Transformer from Scratch — AI Oracle",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Luxury CSS (Minimal — Optimized) ─────────────────────────────────
st.markdown("""
<style>
    .stApp {background-color: #0d1117; color: #e6edf3;}
    [data-testid="stSidebar"] {background: #161b22; border-right: 1px solid #30363d;}
    h1, h2, h3 {color: #e6edf3 !important;}
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 1.2rem; text-align: center;
    }
    .metric-card:hover {border-color: #58a6ff;}
    .metric-value {font-size: 2rem; font-weight: 700; color: #58a6ff;}
    .metric-label {font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem;}
    .gold-accent {color: #d29922; font-weight: 600;}
    .paper-quote {
        border-left: 4px solid #238636; padding: 0.8rem 1.2rem;
        background: #161b22; border-radius: 0 8px 8px 0;
        font-style: italic; color: #8b949e; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model & Data Loading ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_config():
    """Load the trained Transformer model and all configuration files."""
    config_path = MODELS / "config.json"
    model_path  = MODELS / "transformer_classifier.pt"
    label_path  = MODELS / "label_map.json"
    tok_path    = MODELS / "tokenizer.json"

    # Check if all files exist
    missing = [p for p in [config_path, model_path, label_path, tok_path] if not p.exists()]
    if missing:
        return None, None, None, None, [str(m) for m in missing]

    with open(config_path) as f:
        cfg = json.load(f)
    with open(label_path) as f:
        label_data = json.load(f)

    idx2label = {int(k): v for k, v in label_data['idx2label'].items()}

    tokenizer = CharTokenizer()
    tokenizer.load(tok_path)

    model = TransformerClassifier(
        vocab_size  = cfg['vocab_size'],
        d_model     = cfg['d_model'],
        n_layers    = cfg['n_layers'],
        n_heads     = cfg['n_heads'],
        d_ff        = cfg['d_ff'],
        num_classes = cfg['num_classes'],
        max_seq_len = cfg['max_seq_len'],
        dropout     = cfg['dropout'],
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model, cfg, idx2label, tokenizer, []


# ── Load Everything ───────────────────────────────────────────────────
model, cfg, idx2label, tokenizer, missing_files = load_model_and_config()


# ── Add CharTokenizer import path ────────────────────────────────────
# (CharTokenizer is defined in s01_stack_data.py, copied into layers for import)


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 Transformer Oracle")
    st.markdown("---")

    if model is not None:
        st.success("✅ Model Loaded Successfully")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{cfg['n_layers']}</div>
            <div class='metric-label'>Encoder Layers</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{cfg['n_heads']}</div>
            <div class='metric-label'>Attention Heads</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{cfg['d_model']}</div>
            <div class='metric-label'>d_model</div>
        </div>
        """, unsafe_allow_html=True)

        total_params = sum(p.numel() for p in model.parameters())
        st.markdown(f"\n📊 **Parameters**: {total_params:,}")
        st.markdown(f"📦 **Vocab Size**: {cfg['vocab_size']}")
        st.markdown(f"🏷️ **Classes**: {cfg['num_classes']}")
    else:
        st.error("⚠️ Model not trained yet!")
        st.markdown("Run the pipeline first:")
        st.code("python notebooks/s00_pipeline.py", language="bash")
        for f in missing_files:
            st.warning(f"Missing: {f}")


# ── Header ────────────────────────────────────────────────────────────
st.markdown("# 🧠 Transformer from Scratch — Architectural Auditor")
st.markdown("""
<div class='paper-quote'>
"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely."
<br>— Vaswani et al., <i>"Attention is All You Need"</i>, NeurIPS 2017
</div>
""", unsafe_allow_html=True)

# ── Problem & Approach ────────────────────────────────────────────────
col_prob, col_appr = st.columns(2)

with col_prob:
    st.markdown("### 🎯 The Problem")
    st.markdown("""
    **Task**: Given a StackOverflow question (title + body), predict which
    programming language (Python, Java, C++, etc.) the question is about.

    **Why this matters**: This is a real-world **NLP classification** problem.
    Before Transformers, we used RNNs which processed words sequentially (slow!)
    and lost long-range context. The Transformer solves this with **parallel attention**.
    """)

with col_appr:
    st.markdown("### 🛠️ The Approach")
    st.markdown("""
    We build the **complete Transformer Encoder** from scratch — no `nn.Transformer`:

    1. **Character-level tokenizer** (handles code like `pd.DataFrame()`)
    2. **Sinusoidal positional encoding** (the "clock" for word order)
    3. **Multi-Head Self-Attention** (8 parallel heads, 256 dims)
    4. **Position-wise FFN** (nonlinear feature expansion)
    5. **Residual connections + LayerNorm** (gradient highways)
    """)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────
tab_clf, tab_arch, tab_train, tab_attn, tab_paper = st.tabs([
    "🧠 Live Classification",
    "🏗️ Architecture Blueprint",
    "📊 Training Analytics",
    "🔬 Attention Lab",
    "📐 Paper Reference",
])


# ── Tab 1: Live Classification ───────────────────────────────────────
with tab_clf:
    if model is None:
        st.warning("Train the model first: `python notebooks/s00_pipeline.py`")
    else:
        st.markdown("### 🧠 Real-Time Programming Language Classification")

        user_input = st.text_area(
            "Enter a StackOverflow-style question:",
            "How do I merge two DataFrames in pandas using a left join?",
            height=100,
        )

        if st.button("🚀 Classify", type="primary"):
            # Tokenize
            encoded = tokenizer.encode(user_input.lower(), max_len=cfg['max_seq_len'])
            input_tensor = torch.tensor([encoded], dtype=torch.long)

            # Forward pass
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = probs.argmax().item()
                pred_lang = idx2label[pred_idx]
                confidence = probs[pred_idx].item()

            # Results
            col_res, col_shape = st.columns([1, 2])

            with col_res:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{pred_lang}</div>
                    <div class='metric-label'>Predicted Language ({confidence:.1%} confidence)</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Top 5 Predictions")
                top5_vals, top5_idx = probs.topk(5)
                for val, idx in zip(top5_vals, top5_idx):
                    lang = idx2label[idx.item()]
                    pct = val.item()
                    st.progress(pct, text=f"{lang}: {pct:.1%}")

            with col_shape:
                st.markdown("#### 📐 Tensor Shape Flow (Every Stage)")
                st.markdown("*Watch how your input transforms through the Transformer:*")
                shape_df = make_shape_flow_df(model.encoder.shape_log)
                st.dataframe(shape_df, use_container_width=True, hide_index=True)

                st.info(
                    "💡 **Key Insight**: Notice the shape stays `[1, seq_len, 256]` "
                    "through all encoder layers. This is the power of **residual connections** — "
                    "they preserve dimensionality while enriching the representation."
                )


# ── Tab 2: Architecture Blueprint ────────────────────────────────────
with tab_arch:
    st.markdown("### 🏗️ Complete Transformer Encoder Blueprint")

    col_diag, col_explain = st.columns([1, 1])

    with col_diag:
        n_layers = cfg['n_layers'] if cfg else 3
        st.plotly_chart(plot_architecture_diagram(n_layers), use_container_width=True)

    with col_explain:
        st.markdown("""
        #### Stage-by-Stage Breakdown

        **1. Input Token IDs** `(batch, seq_len)`
        Raw character indices from our custom tokenizer.
        Each character maps to a unique integer (vocab ~100).

        **2. Token Embedding** `(batch, seq_len, d_model)`
        Each integer becomes a 256-dimensional dense vector.
        Scaled by √d_model to balance with positional encodings.

        **3. Positional Encoding** `(batch, seq_len, d_model)`
        Sinusoidal waves are **added** (not concatenated!) to embeddings.
        Even dims use `sin()`, odd dims use `cos()`.
        This gives the model its "sense of order."

        **4. Encoder Layers** `(batch, seq_len, d_model)`
        Each layer applies:
        - **Multi-Head Self-Attention**: Every token "looks at" every other token
        - **Add & LayerNorm**: Residual skip + normalization
        - **Feed-Forward Network**: 256 → 1024 → 256 (expand then contract)
        - **Add & LayerNorm**: Another residual skip

        **5. Mean Pooling** `(batch, d_model)`
        Average all token representations into one vector.

        **6. Classification** `(batch, num_classes)`
        Final linear layer + softmax for prediction.
        """)


# ── Tab 3: Training Analytics ────────────────────────────────────────
with tab_train:
    st.markdown("### 📊 Training Performance Dashboard")

    history_path = MODELS / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        # KPI Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{history['val_acc'][-1]:.1%}</div>
                <div class='metric-label'>Final Val Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{min(history['val_loss']):.4f}</div>
                <div class='metric-label'>Best Val Loss</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(history['train_loss'])}</div>
                <div class='metric-label'>Epochs Trained</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.plotly_chart(plot_training_history(history), use_container_width=True)
    else:
        st.warning("No training history found. Run the pipeline first.")


# ── Tab 4: Attention Lab ─────────────────────────────────────────────
with tab_attn:
    st.markdown("### 🔬 Multi-Head Attention Exploration")

    if model is None:
        st.warning("Train the model first.")
    else:
        attn_input = st.text_input(
            "Enter text to analyze attention patterns:",
            "How to sort a list in Python using lambda functions?"
        )

        if attn_input:
            encoded = tokenizer.encode(attn_input.lower(), max_len=cfg['max_seq_len'])
            input_tensor = torch.tensor([encoded], dtype=torch.long)

            with torch.no_grad():
                _ = model(input_tensor)

            # Get attention from last encoder layer
            last_attn = model.encoder.layers[-1].self_attn.attn_weights[0]

            # Show first 30 positions for readability
            show_len = min(30, len(attn_input))
            char_labels = list(attn_input[:show_len])
            trimmed_attn = last_attn[:, :show_len, :show_len]

            head_idx = st.slider("Select Attention Head", 0, cfg['n_heads'] - 1, 0)
            st.plotly_chart(
                plot_attention_heatmap(trimmed_attn, head_idx, char_labels),
                use_container_width=True,
            )

            st.markdown("""
            **How to read this heatmap:**
            - Each row is a **Query** character (the one "asking")
            - Each column is a **Key** character (the one "answering")
            - Bright spots mean high attention ("this character is important for understanding me")
            - A perfect diagonal means "each character attends most to itself"
            - Off-diagonal patterns reveal **learned relationships** between characters
            """)

        # Positional Encoding
        st.markdown("---")
        st.markdown("### 🗺️ Positional Encoding Visualization")
        st.markdown("""
        Since Transformers have no recurrence (no RNN), they don't know word order.
        These sinusoidal waves inject a unique "coordinate" into each position.
        Notice how each row (position) has a distinct pattern — this is the model's "clock."
        """)
        st.plotly_chart(plot_positional_encoding(d_model=(cfg or {}).get('d_model', 256)),
                        use_container_width=True)


# ── Tab 5: Paper Reference ───────────────────────────────────────────
with tab_paper:
    st.markdown("### 📐 \"Attention is All You Need\" — Mathematical Reference")
    st.markdown("""
    <div class='paper-quote'>
    This section contains the exact mathematical formulations from the original 2017 paper,
    mapped to the code in this project.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 1. Scaled Dot-Product Attention")
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    st.markdown("""
    **In our code** → `notebooks/layers.py : ScaledDotProductAttention`

    - **Q (Query)**: "What am I looking for?" — projected from input via `W_Q`
    - **K (Key)**: "What information do I contain?" — projected via `W_K`
    - **V (Value)**: "What should I pass forward?" — projected via `W_V`
    - **√d_k**: Scaling factor prevents dot products from growing too large,
      which would push softmax into regions with vanishing gradients.
    """)

    st.markdown("---")
    st.markdown("#### 2. Multi-Head Attention")
    st.latex(r"\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W^O")
    st.latex(r"\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)")
    st.markdown(f"""
    **In our code** → `notebooks/layers.py : MultiHeadAttention`

    We use **h = {(cfg or {}).get('n_heads', 8)} heads**, splitting d_model = {(cfg or {}).get('d_model', 256)}
    into {(cfg or {}).get('n_heads', 8)} subspaces of d_k = {(cfg or {}).get('d_model', 256) // (cfg or {}).get('n_heads', 8)} each.

    **Why multiple heads?**
    Each head can learn different linguistic patterns:
    - Head 1 might learn "syntax relationships" (brackets match)
    - Head 2 might learn "keyword proximity" (import → library_name)
    - Head 3 might learn "code structure" (function → parameters)
    """)

    st.markdown("---")
    st.markdown("#### 3. Positional Encoding")
    st.latex(r"PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)")
    st.latex(r"PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)")
    st.markdown("""
    **In our code** → `notebooks/layers.py : PositionalEncoding`

    **Why sinusoidal (not learned)?**
    1. Can extrapolate to sequence lengths never seen during training.
    2. `PE(pos+k)` can be represented as a linear function of `PE(pos)`,
       so the model can easily learn to attend to relative positions.
    3. No extra parameters needed (register_buffer, not nn.Parameter).
    """)

    st.markdown("---")
    st.markdown("#### 4. Residual Connections + Layer Normalization")
    st.latex(r"\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))")
    st.markdown("""
    **In our code** → `model.py : EncoderLayer`

    **Why residual connections?**
    Without them, gradients would vanish before reaching early layers in deep networks.
    The skip connection creates a **"gradient highway"** that lets information flow
    directly from layer 1 to layer N without degradation.

    **Why LayerNorm (not BatchNorm)?**
    LayerNorm normalizes across features for each sample independently,
    making it more stable for variable-length sequences and small batch sizes.
    """)

    st.markdown("---")
    st.markdown("#### 5. Position-Wise Feed-Forward Network")
    st.latex(r"\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2")
    st.markdown(f"""
    **In our code** → `notebooks/layers.py : PositionWiseFeedForward`

    - **Expansion**: d_model ({cfg.get('d_model', 256)}) → d_ff ({cfg.get('d_ff', 1024)}) — 4× wider
    - **ReLU activation**: Introduces nonlinearity
    - **Contraction**: d_ff ({cfg.get('d_ff', 1024)}) → d_model ({cfg.get('d_model', 256)}) — back to original

    This "expand-and-contract" pattern gives each token independent "thinking capacity"
    after the attention step has mixed information across positions.
    """)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Transformer from Scratch · Built with PyTorch · "
    "Paper: Vaswani et al., 'Attention is All You Need', NeurIPS 2017 · "
    "Dataset: StackOverflow Questions 2020-2025"
)
