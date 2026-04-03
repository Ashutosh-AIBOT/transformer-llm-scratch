---
title: Transformer LLM
emoji: 🚀
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 Transformer from Scratch — Full Architecture Deep Dive

Build the complete **Transformer Encoder** from scratch using PyTorch,
trained on 95K StackOverflow questions to classify programming languages.

> *"We propose a new simple network architecture, the Transformer,
> based solely on attention mechanisms, dispensing with recurrence
> and convolutions entirely."*
> — Vaswani et al., "Attention is All You Need", NeurIPS 2017

---

## 📐 Architecture (Paper-Accurate)

| Component | Formula / Logic | File |
|:---|:---|:---|
| **Scaled Dot-Product Attention** | `softmax(QK^T / √d_k) × V` | `notebooks/layers.py` |
| **Multi-Head Attention** | `Concat(head_1...head_h) × W^O` | `notebooks/layers.py` |
| **Positional Encoding** | `sin(pos / 10000^(2i/d))`, `cos(...)` | `notebooks/layers.py` |
| **Feed-Forward Network** | `max(0, xW₁ + b₁)W₂ + b₂` | `notebooks/layers.py` |
| **Add & LayerNorm** | `LayerNorm(x + SubLayer(x))` | `model.py` |
| **Token Embedding** | `Embedding(x) × √d_model` | `notebooks/layers.py` |

## 🧬 Shape Flow

```
Token IDs:        (batch, seq_len)           e.g., (32, 128)
→ Embedding:      (batch, seq_len, d_model)  e.g., (32, 128, 256)
→ + PosEnc:       (batch, seq_len, d_model)  e.g., (32, 128, 256)
→ Encoder ×3:     (batch, seq_len, d_model)  e.g., (32, 128, 256)
→ Mean Pool:      (batch, d_model)           e.g., (32, 256)
→ Linear Head:    (batch, num_classes)       e.g., (32, 20)
```

## 🚀 Getting Started

```bash
# 1. Run the full pipeline (data + training)
cd "Transformer from Scratch"
conda run -n agent python notebooks/s00_pipeline.py

# 2. Launch the dashboard
conda run -n agent streamlit run app.py
```

## 📁 Project Structure
```
Transformer from Scratch/
├── model.py               # Full Transformer (Encoder + Classifier)
├── app.py                 # Luxury 5-tab Streamlit dashboard
├── viz_engine.py           # Plotly-based visualizations
├── path_utils.py           # Centralized path management
├── notebooks/
│   ├── s00_pipeline.py     # Master pipeline orchestrator
│   ├── s01_stack_data.py   # Data engineering + tokenizer
│   ├── s02_train.py        # Training loop (AMP + tqdm)
│   └── layers.py           # All Transformer sub-layers
├── data/raw/               # StackOverflow CSV
├── models/                 # Trained weights + config
└── charts/                 # Saved visualizations
```

## 🎯 Dashboard Tabs
1. **🧠 Live Classification** — Predict programming language + see tensor shapes
2. **🏗️ Architecture Blueprint** — Visual encoder flow chart
3. **📊 Training Analytics** — Loss curves & accuracy metrics
4. **🔬 Attention Lab** — Multi-head attention heatmaps
5. **📐 Paper Reference** — Complete LaTeX formulas from the original paper
