"""
prepare_dataset.py
==================
Converts the flat binary DeepPCB dataset into unified YOLO format under data/yolo_dataset/.

Directory layout produced:
  data/yolo_dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/

Source data:
  data/good/ (images only)
  data/bad/  (images only)

Class mapping (matches dataset.yaml):
  0: GOOD
  1: DEFECTIVE
"""

import os
import shutil
import random
from pathlib import Path
import cv2

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA_DIR   = BASE / "data"
OUT_DIR    = DATA_DIR / "yolo_dataset"
GOOD_DIR   = DATA_DIR / "good"
BAD_DIR    = DATA_DIR / "bad"

VAL_RATIO  = 0.15   # 15 % held out for validation
SEED       = 42
random.seed(SEED)

# ── YOLO class IDs ────────────────────────────────────────────────────────────
CLS_GOOD = 0
CLS_BAD  = 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directory created/reset: {OUT_DIR}")


def copy_and_label(src_img, dst_split, stem, labels):
    """Copy image to dest, write YOLO label file."""
    img_dst = OUT_DIR / "images" / dst_split / (stem + Path(src_img).suffix)
    lbl_dst = OUT_DIR / "labels" / dst_split / (stem + ".txt")
    shutil.copy2(src_img, img_dst)
    with open(lbl_dst, "w") as f:
        for line in labels:
            f.write(line + "\n")


def split(items):
    """Return (train_list, val_list) with fixed seed."""
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    return items[n_val:], items[:n_val]


# ── Dataset Processing ────────────────────────────────────────────────────────
def process_binary():
    print("\nProcessing GOOD and BAD folders …")
    
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    good_imgs = [f for f in GOOD_DIR.iterdir() if f.suffix.lower() in EXTS]
    bad_imgs  = [f for f in BAD_DIR.iterdir() if f.suffix.lower() in EXTS]
    
    # Shuffle right away to get a random mix, then limit to max 500 each
    random.shuffle(good_imgs)
    random.shuffle(bad_imgs)
    good_imgs = good_imgs[:500]
    bad_imgs  = bad_imgs[:500]
    
    if not good_imgs and not bad_imgs:
        print("❌ No images found in data/good or data/bad.")
        return

    # Create dummy bounding box for the entire image (cx, cy, w, h = 0.5)
    def _proc(images, cls_id, prefix):
        train_imgs, val_imgs = split(list(images))
        
        for imgs, split_name in [(train_imgs, "train"), (val_imgs, "val")]:
            for img_p in imgs:
                # Assuming full image is the object for simplified binary classification
                # CX, CY, W, H = 0.5, 0.5, 1.0, 1.0
                labels = [f"{cls_id} 0.500000 0.500000 1.000000 1.000000"]
                stem = f"{prefix}_{img_p.stem}"
                copy_and_label(img_p, split_name, stem, labels)
                
        return len(train_imgs), len(val_imgs)

    g_train, g_val = _proc(good_imgs, CLS_GOOD, "ok")
    b_train, b_val = _proc(bad_imgs, CLS_BAD, "ng")
    
    print(f"   → GOOD: {g_train} train / {g_val} val")
    print(f"   → BAD:  {b_train} train / {b_val} val")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Defect Detection — Dataset Preparation (GOOD/BAD)")
    print("=" * 60)

    make_dirs()
    process_binary()

    # Summary
    for split in ("train", "val"):
        if (OUT_DIR / "images" / split).exists():
            n = len(list((OUT_DIR / "images" / split).glob("*")))
            print(f"\n  {split}: {n} images")

    print("\n✅ Dataset preparation complete!")
    print(f"   Output → {OUT_DIR}")
    print("   Run: python train.py")
