"""
model_service.py
================
YOLOv8 inference service for the AI Defect Detection Flask app.

Supports the 10-class unified model trained on:
  Bridge_Crack_Image | CrackForest | DeepPCB | Magnetic-Tile-Defect
"""

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os

# ── Class metadata (10-class DeepPCB model) ──────────────────────────────────
CLASS_INFO = {
    0:  {"name": "crack",     "color": (255,  61,  90),  "source": "DeepPCB"},
    1:  {"name": "blowhole",  "color": (255, 165,   0),  "source": "DeepPCB"},
    2:  {"name": "break",     "color": (255, 255,   0),  "source": "DeepPCB"},
    3:  {"name": "fray",      "color": (  0, 255,   0),  "source": "DeepPCB"},
    4:  {"name": "open",      "color": (  0, 255, 255),  "source": "DeepPCB"},
    5:  {"name": "short",     "color": (  0, 100, 255),  "source": "DeepPCB"},
    6:  {"name": "mousebite", "color": (255,   0, 255),  "source": "DeepPCB"},
    7:  {"name": "spur",      "color": (128,   0, 128),  "source": "DeepPCB"},
    8:  {"name": "copper",    "color": (139,  69,  19),  "source": "DeepPCB"},
    9:  {"name": "pin_hole",  "color": (192, 192, 192),  "source": "DeepPCB"},
}

MODEL_PATH   = "models/best.pt"
FALLBACK_PT  = "yolov8n.pt"
CONF_THRESH  = 0.40   # minimum detection confidence


def load_model():
    """Load the custom-trained model, falling back to YOLOv8n if not found."""
    if os.path.exists(MODEL_PATH):
        print(f"✓ Loading custom model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    else:
        print(f"⚠️  Custom model not found at '{MODEL_PATH}'")
        print(f"   → Using fallback: {FALLBACK_PT}")
        print("   To train:  python prepare_dataset.py && python train.py")
        model = YOLO(FALLBACK_PT)
    return model


def predict_defects(model, image_bytes):
    """
    Run detection on raw image bytes.

    Returns
    -------
    dict with keys:
        status          : 'PASS' or 'FAIL'
        defects         : list of detection dicts
        confidence      : max confidence across all detections
        annotated_image : base64-encoded annotated PNG
        model_info      : info about which model was used
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")

    # Inference
    results = model(img, conf=CONF_THRESH, verbose=False)[0]

    defects  = []
    max_conf = 0.0

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = r
        cls_id   = int(class_id)
        cls_meta = CLASS_INFO.get(cls_id, {"name": model.names.get(cls_id, "Unknown"), "source": "-"})
        defects.append({
            "class":      cls_meta["name"],
            "class_id":   cls_id,
            "confidence": round(float(conf), 4),
            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            "source":     cls_meta.get("source", "-"),
        })
        max_conf = max(max_conf, conf)

    # Sort by confidence descending
    defects.sort(key=lambda d: d["confidence"], reverse=True)

    status = "PASS" if len(defects) == 0 else "FAIL"

    # Annotate
    annotated = results.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(annotated)
    buf       = BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64   = base64.b64encode(buf.getvalue()).decode()

    # Determine which model is active
    is_custom = os.path.exists(MODEL_PATH)
    model_info = {
        "type":   "Custom (trained)" if is_custom else "Pretrained YOLOv8n (fallback)",
        "path":   MODEL_PATH if is_custom else FALLBACK_PT,
        "classes": len(CLASS_INFO),
    }

    return {
        "status":          status,
        "defects":         defects,
        "confidence":      round(float(max_conf), 4) if max_conf > 0 else 0.0,
        "annotated_image": img_b64,
        "model_info":      model_info,
    }
