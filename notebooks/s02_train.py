"""
========================================================================
STOP 02 — TRAINING PIPELINE (GTX 1650 Optimized)
========================================================================
Trains the Transformer Classifier on StackOverflow data.

Optimizations for GTX 1650 (4GB VRAM):
  - Mixed Precision Training (torch.amp) → ~2x speedup
  - Gradient Accumulation (optional)
  - Small batch size (32) to stay under 3.5GB VRAM
  - tqdm progress bars for every epoch and batch

Training Strategy:
  - Optimizer: AdamW (weight decay for regularization)
  - Scheduler: OneCycleLR (warm-up then decay)
  - Epochs: 15 (tuned for 2-hour limit on GTX 1650)
========================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import json
import time
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TransformerClassifier
from path_utils import DATA_PROCESSED, MODELS, CHARTS


def train_model(epochs=15, batch_size=32, lr=3e-4):
    """
    Full training loop with:
      - Mixed precision (AMP) for GPU efficiency
      - tqdm progress bars (per-epoch and per-batch)
      - Training history saved for dashboard visualization
    """
    print("=" * 60)
    print("STOP 02 — TRANSFORMER TRAINING PIPELINE")
    print("=" * 60)

    # ── Device Selection ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")
        print(f"[VRAM]   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Load Processed Data ───────────────────────────────────────────
    print("\n[1/5] Loading processed data...")
    X_train = torch.load(DATA_PROCESSED / "X_train.pt")
    X_val   = torch.load(DATA_PROCESSED / "X_val.pt")
    y_train = torch.load(DATA_PROCESSED / "y_train.pt")
    y_val   = torch.load(DATA_PROCESSED / "y_val.pt")

    print(f"       Train: {X_train.shape}")
    print(f"       Val:   {X_val.shape}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Load Config & Build Model ─────────────────────────────────────
    print("\n[2/5] Building Transformer model...")
    with open(MODELS / "config.json", 'r') as f:
        cfg = json.load(f)

    model = TransformerClassifier(
        vocab_size   = cfg['vocab_size'],
        d_model      = cfg['d_model'],
        n_layers     = cfg['n_layers'],
        n_heads      = cfg['n_heads'],
        d_ff         = cfg['d_ff'],
        num_classes  = cfg['num_classes'],
        max_seq_len  = cfg['max_seq_len'],
        dropout      = cfg['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {total_params:,}")
    print(f"       Config: d_model={cfg['d_model']}, heads={cfg['n_heads']}, "
          f"layers={cfg['n_layers']}, d_ff={cfg['d_ff']}")

    # ── Training Setup ────────────────────────────────────────────────
    print("\n[3/5] Setting up optimizer & scheduler...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    # ── Training Loop ─────────────────────────────────────────────────
    print(f"\n[4/5] Training for {epochs} epochs...")
    print("-" * 60)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        start_time = time.time()

        # ── Train Phase ───────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{epochs} [TRAIN]",
                     leave=False, ncols=80)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            if scaler:
                with autocast("cuda"):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / n_batches

        # ── Validation Phase ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if device.type == "cuda":
                    with autocast("cuda"):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                else:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        elapsed = time.time() - start_time

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1:2d}/{epochs} │ "
              f"Train Loss: {avg_train_loss:.4f} │ "
              f"Val Loss: {avg_val_loss:.4f} │ "
              f"Val Acc: {val_acc:.4f} │ "
              f"Time: {elapsed:.1f}s")

    # ── Save Model & History ──────────────────────────────────────────
    print(f"\n[5/5] Saving model and training history...")
    torch.save(model.state_dict(), MODELS / "transformer_classifier.pt")

    with open(MODELS / "training_history.json", 'w') as f:
        json.dump(history, f)

    print("\n" + "=" * 60)
    print("✅ STOP 02 COMPLETE — Training finished!")
    print(f"   Model:   {MODELS / 'transformer_classifier.pt'}")
    print(f"   History: {MODELS / 'training_history.json'}")
    print(f"   Best Val Accuracy: {max(history['val_acc']):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
