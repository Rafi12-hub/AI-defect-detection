"""
incremental_retrain.py
======================
Fine-tunes the latest `best.pt` model on newly labeled data when the trigger threshold is met.
"""

import os
import yaml
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

BASE = Path(__file__).parent.parent
SETTINGS_FILE = BASE / "config" / "settings.yaml"

with open(SETTINGS_FILE, "r") as f:
    settings = yaml.safe_load(f)

# Configs
increment_settings = settings['training']['increment']
MODEL_OUT = BASE / "models" / "best.pt"
NEWLY_LABELED_DIR = BASE / settings['active_learning']['labeled_dir']

epochs = increment_settings['epochs']
batch_size = increment_settings['batch_size']
PROJECT = str(BASE / "runs" / "detect")
RUN_NAME = "incremental_retrain"
DEVICE = "0" if torch.cuda.is_available() else "cpu"

def create_incremental_dataset_yaml():
    """
    Creates a temporary YAML file specifically for the newly labeled dataset.
    """
    # Read from the source dataset.yaml to preserve correct class mapping
    YAML = BASE / "dataset.yaml"
    try:
        with open(YAML, "r") as f:
            src = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        src = {"nc": 10, "names": {
            0: "crack", 1: "blowhole", 2: "break", 3: "fray", 4: "open",
            5: "short", 6: "mousebite", 7: "spur", 8: "copper", 9: "pin_hole",
        }}

    abs_yaml = {
        "path": str(NEWLY_LABELED_DIR).replace("\\", "/"),
        "train": ".",  # Since all images and txts are in the same folder
        "val": ".",    # We use the same for validation in this tiny set
        "nc": src.get("nc", 10),
        "names": src.get("names", {0: "GOOD", 1: "DEFECTIVE"}),
    }
    tmp_yaml = BASE / "incremental_dataset.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(abs_yaml, f, default_flow_style=False, sort_keys=False)
    return str(tmp_yaml)


def run_incremental_retraining():
    print(f"\n{'='*60}")
    print(f"  AI Defect Detection — Incremental Retraining (Active Learning)")
    print(f"{'='*60}\n")
    
    labeled_images = list(NEWLY_LABELED_DIR.glob("*.jpg"))
    if not labeled_images:
        print("❌ No newly labeled images available for retraining.")
        return False
        
    print(f"Found {len(labeled_images)} newly labeled images. Starting fine-tuning...")
    
    # Generate yaml
    dataset_yaml = create_incremental_dataset_yaml()
    
    # Load current best model
    if not MODEL_OUT.exists():
        print(f"❌ Current production model {MODEL_OUT} not found!")
        return False
        
    model = YOLO(str(MODEL_OUT))
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        project=PROJECT,
        name=RUN_NAME,
        device=DEVICE,
        exist_ok=True,
        lr0=0.0001, # lower learning rate for fine-tuning
        lrf=0.01,
        warmup_epochs=0 # No warmup
    )
    
    # Save best new model
    best_weights = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        # Backup old best model
        backup_model = MODEL_OUT.parent / "best_backup.pt"
        shutil.copy2(MODEL_OUT, backup_model)
        
        # Replace with new best model
        shutil.copy2(best_weights, MODEL_OUT)
        print(f"\n✅ Successfully fine-tuned and replaced best weights at {MODEL_OUT}")
        print("   Previous model backed up to models/best_backup.pt")
        
        # Clear newly_labeled since it has been consumed
        # NOTE: In production, moving them to a 'consumed_data' might be better,
        # but for this MVP, we clear them to reset the trigger.
        consumed_dir = BASE / "consumed_labeled_data"
        consumed_dir.mkdir(exist_ok=True)
        for f in NEWLY_LABELED_DIR.iterdir():
            if f.is_file():
                shutil.move(str(f), str(consumed_dir / f.name))
        print(f"🧹 Cleaned up {len(labeled_images)} files from newly_labeled/")
    else:
        print(f"\n⚠️  Could not find weights at {best_weights} after training.")
        
    return True

if __name__ == "__main__":
    run_incremental_retraining()
