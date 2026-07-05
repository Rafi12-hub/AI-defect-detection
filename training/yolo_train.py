"""
yolo_train.py
=============
Train YOLOv8 on the unified defect dataset with Transfer Learning and Augmentations.

USAGE
-----
  python training/yolo_train.py
"""

import os
import shutil
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
YAML       = BASE / "dataset.yaml"
MODEL_OUT  = BASE / "models" / "best.pt"
SETTINGS_FILE = BASE / "config" / "settings.yaml"

# Load settings
with open(SETTINGS_FILE, "r") as f:
    settings = yaml.safe_load(f)

img_size   = settings['training']['yolo']['img_size']
batch_size = settings['training']['yolo']['batch_size']
epochs     = settings['training']['yolo']['epochs']
workers    = settings['training']['yolo']['workers']
patience   = settings['training']['yolo']['patience']
freeze     = settings['training']['yolo'].get('freeze', None)
model_base = settings['training']['yolo']['base_model']

aug_settings = settings.get('augmentations', {}).get('yolo', {})

PROJECT    = str(BASE / "runs" / "detect")
RUN_NAME   = "defect_detector"
DEVICE     = "0" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset():
    """Verify that dataset images exist."""
    dataset_dir = BASE / "data" / "yolo_dataset"
    train_imgs  = dataset_dir / "images" / "train"
    val_imgs    = dataset_dir / "images" / "val"

    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        print("❌ Training images not found.")
        return False

    n_train = len(list(train_imgs.glob("*")))
    n_val   = len(list(val_imgs.glob("*"))) if val_imgs.exists() else 0
    print(f"✅ Dataset ready  →  {n_train} train / {n_val} val images")
    return True

def make_abs_yaml():
    # Read from the source dataset.yaml to preserve correct class mapping
    with open(YAML, "r") as f:
        src = yaml.safe_load(f)
    dataset_root = (BASE / "data" / "yolo_dataset").resolve()
    abs_yaml = {
        "path":  str(dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    src.get("nc", 2),
        "names": src.get("names", {0: "GOOD", 1: "DEFECTIVE"}),
    }
    tmp_yaml = BASE / "dataset_abs.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(abs_yaml, f, default_flow_style=False, sort_keys=False)
    return str(tmp_yaml)


def train():
    if not check_dataset():
        return

    abs_yaml_path = make_abs_yaml()

    print(f"\n{'='*60}")
    print(f"  AI Defect Detection — YOLOv8 Transfer Learning")
    print(f"{'='*60}")
    
    actual_model_base = model_base
    if not os.path.exists(BASE / actual_model_base):
        print(f"⚠️  Base model {actual_model_base} not found, falling back to yolov8n.pt")
        actual_model_base = "yolov8n.pt"

    print(f"  Base model : {actual_model_base}")
    print(f"  Epochs     : {epochs}  (patience={patience})")
    print(f"  Device     : {DEVICE}")
    print(f"  Freeze     : {freeze} layers")
    print(f"{'='*60}\n")

    # Load pre-trained YOLOv8 model
    model = YOLO(str(BASE / actual_model_base) if actual_model_base != "yolov8n.pt" else actual_model_base)

    train_args = {
        "data": abs_yaml_path,
        "epochs": epochs,
        "imgsz": img_size,
        "batch": batch_size,
        "workers": workers,
        "patience": patience,
        "project": PROJECT,
        "name": RUN_NAME,
        "device": DEVICE,
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
    }

    if freeze is not None:
        train_args["freeze"] = int(freeze)

    # Apply augmentations from config
    for k, v in aug_settings.items():
        train_args[k] = v

    results = model.train(**train_args)

    best_weights = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        MODEL_OUT.parent.mkdir(exist_ok=True)
        shutil.copy2(best_weights, MODEL_OUT)
        print(f"\n✅ Best weights saved → {MODEL_OUT}")
    else:
        print(f"\n⚠️  Could not find weights at {best_weights}")

    return results

if __name__ == "__main__":
    train()
