"""
========================================================================
MASTER PIPELINE — Run All Steps Sequentially
========================================================================
Usage:  python notebooks/s00_pipeline.py

This orchestrates the full Transformer from Scratch workflow:
  STOP 01 → Data Engineering (clean, tokenize, split)
  STOP 02 → Model Training (15 epochs with AMP)

After this completes, launch the dashboard:
  streamlit run app.py
========================================================================
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("╔" + "═" * 58 + "╗")
    print("║   TRANSFORMER FROM SCRATCH — MASTER PIPELINE             ║")
    print("╚" + "═" * 58 + "╝")

    # ── STOP 01: Data Engineering ─────────────────────────────────────
    print("\n" + "▶" * 30 + " STOP 01: DATA ENGINEERING")
    from notebooks.s01_stack_data import run_data_pipeline
    run_data_pipeline()

    # ── STOP 02: Training ─────────────────────────────────────────────
    print("\n" + "▶" * 30 + " STOP 02: MODEL TRAINING")
    from notebooks.s02_train import train_model
    train_model()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   ✅ ALL STOPS COMPLETE — Ready for Dashboard!            ║")
    print("║   Run: streamlit run app.py                              ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
