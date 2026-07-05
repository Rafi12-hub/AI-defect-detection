"""
cnn_lstm_train.py
=================
Train the CNN + LSTM binary defect classifier.

USAGE
-----
  # First reorganise datasets (one-time):
  python reorganize_dataset.py

  # Then train:
  python cnn_lstm_train.py
  python cnn_lstm_train.py --epochs 5          # quick smoke test
  python cnn_lstm_train.py --epochs 30 --lr 5e-4

Outputs
-------
  models/cnn_lstm_best.pth  — best weights (by val accuracy)
  models/cnn_lstm_last.pth  — final epoch weights
"""

import argparse
import os
import json
import random
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from cnn_lstm_model import build_model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
GOOD_DIR   = BASE / "data" / "good"
BAD_DIR    = BASE / "data" / "bad"
MODELS_DIR = BASE / "models"
BEST_PT    = MODELS_DIR / "cnn_lstm_best.pth"
LAST_PT    = MODELS_DIR / "cnn_lstm_last.pth"

# ── Default hyper-params ───────────────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 16
LR          = 1e-4
VAL_SPLIT   = 0.20
IMG_SIZE    = 224
SEED        = 42


# ── Dataset ───────────────────────────────────────────────────────────────────
class DefectDataset(Dataset):
    """Binary image dataset: 0 = good, 1 = defective (bad)."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, samples: list, transform=None):
        """
        Parameters
        ----------
        samples : list of (path_str, label_int) tuples
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def load_samples() -> list:
    """Collect (path, label) tuples from good/ and bad/ dirs."""
    samples = []
    exts = DefectDataset.EXTS

    def _gather(dir_: Path, label: int):
        count = 0
        for f in dir_.glob("*"):
            if f.suffix.lower() in exts:
                samples.append((str(f), label))
                count += 1
        return count

    if not GOOD_DIR.exists() or not BAD_DIR.exists():
        raise FileNotFoundError(
            "data/good/ or data/bad/ not found.\n"
            "Run:  python reorganize_dataset.py  first."
        )

    n_good = _gather(GOOD_DIR, 0)
    n_bad  = _gather(BAD_DIR,  1)
    print(f"  Loaded {n_good} GOOD + {n_bad} BAD = {len(samples)} total images")
    return samples


def split_samples(samples, val_ratio=VAL_SPLIT, seed=SEED):
    random.seed(seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_ratio))
    return samples[n_val:], samples[:n_val]


# ── Training loop ─────────────────────────────────────────────────────────────
def train(epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Data
    all_samples = load_samples()
    train_s, val_s = split_samples(all_samples)

    train_ds = DefectDataset(train_s, TRAIN_TF)
    val_ds   = DefectDataset(val_s,   VAL_TF)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    # Model
    model = build_model(pretrained=True).to(device)

    # Optimiser + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    MODELS_DIR.mkdir(exist_ok=True)

    best_val_acc  = 0.0
    history       = []

    print(f"\n{'='*60}")
    print(f"  CNN + LSTM Defect Classifier Training")
    print(f"{'='*60}")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"  Epochs: {epochs}   LR: {lr}   Batch: {batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time()

        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += ((preds > 0.5).float() == labels).sum().item()

        scheduler.step()

        # ── Val ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                preds = model(imgs)
                loss  = criterion(preds, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += ((preds > 0.5).float() == labels).sum().item()

        # ── Stats ─────────────────────────────────────────────────
        n_train = len(train_ds)
        n_val   = len(val_ds)
        train_acc = train_correct / n_train * 100
        val_acc   = val_correct   / n_val   * 100
        elapsed   = time() - t0

        print(f"  Epoch [{epoch:03d}/{epochs}]  "
              f"Train loss: {train_loss/n_train:.4f}  acc: {train_acc:5.1f}%  |  "
              f"Val loss: {val_loss/n_val:.4f}  acc: {val_acc:5.1f}%  "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss / n_train, 4),
            "train_acc":  round(train_acc, 2),
            "val_loss":   round(val_loss / n_val, 4),
            "val_acc":    round(val_acc, 2),
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(BEST_PT))
            print(f"    ✅ New best val accuracy: {best_val_acc:.1f}%  → saved to {BEST_PT}")

    # Save last
    torch.save(model.state_dict(), str(LAST_PT))

    # Save history
    hist_path = MODELS_DIR / "cnn_lstm_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy  : {best_val_acc:.1f}%")
    print(f"  Best weights saved : {BEST_PT}")
    print(f"  Training history   : {hist_path}")
    print(f"{'='*60}")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Defect Classifier")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
