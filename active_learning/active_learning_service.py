"""
active_learning_service.py
==========================
Service to capture low-confidence images and handle reviewed/labeled ones.
"""

import os
import yaml
import shutil
import cv2
import numpy as np
import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
SETTINGS_FILE = BASE / "config" / "settings.yaml"

try:
    with open(SETTINGS_FILE, "r") as f:
        settings = yaml.safe_load(f) or {}
except (FileNotFoundError, yaml.YAMLError):
    settings = {}

al_settings = settings.get('active_learning', {})
CONFIDENCE_THRESHOLD = al_settings.get('confidence_threshold', 0.80)
TRIGGER_COUNT = al_settings.get('retrain_trigger_count', 50)

# Output directories
PENDING_DIR = BASE / al_settings.get('pending_dir', 'review_pending')
LABELED_DIR = BASE / al_settings.get('labeled_dir', 'newly_labeled')

# Ensure directories exist
PENDING_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DIR.mkdir(parents=True, exist_ok=True)

def handle_detection(image_bytes: bytes, confidence: float, class_id: int, has_defects: bool = False):
    """
    Called after YOLO inference. If confidence < threshold and there ARE defects,
    saves the image to review_pending/.
    """
    if has_defects and confidence < CONFIDENCE_THRESHOLD:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"review_pending_{timestamp}_conf{confidence:.2f}_cls{class_id}.jpg"
            filepath = PENDING_DIR / filename
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            cv2.imwrite(str(filepath), img)
            return {"status": "FLAGGED", "file": filename, "reason": "Low confidence score"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
            
    return {"status": "OK"}

def get_pending_images():
    """List images waiting for review."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f.name for f in PENDING_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

def label_image(filename: str, label_class: str, bounding_boxes: list = None):
    """
    Labels an image from review_pending/ and moves it to newly_labeled/.
    Bounding boxes can be saved in a YOLO format txt file.
    """
    pending_path = PENDING_DIR / filename
    if not pending_path.exists():
        return {"error": "File not found"}

    try:
        # Move image
        labeled_img_path = LABELED_DIR / filename
        shutil.move(str(pending_path), str(labeled_img_path))
        
        # Save YOLO annotation file for fine-tuning
        if bounding_boxes is not None:
            label_filename = filename.rsplit(".", 1)[0] + ".txt"
            label_filepath = LABELED_DIR / label_filename
            with open(label_filepath, "w") as f:
                for box in bounding_boxes:
                    # format: class_id center_x center_y width height
                    f.write(f"{box['class_id']} {box['cx']} {box['cy']} {box['w']} {box['h']}\n")
                    
        return {"status": "SUCCESS"}
    except Exception as e:
        return {"error": str(e)}

def check_retrain_trigger():
    """Check if the number of newly labeled images exceeds the threshold."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    labeled_count = len([f for f in LABELED_DIR.iterdir()
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
    return labeled_count >= TRIGGER_COUNT
