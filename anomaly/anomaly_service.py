"""
anomaly_service.py
==================
Inference service for the CNN Autoencoder to detect anomalies.
"""

import os
import yaml
import numpy as np
from pathlib import Path
import cv2

BASE = Path(__file__).parent.parent
SETTINGS_FILE = BASE / "config" / "settings.yaml"

try:
    with open(SETTINGS_FILE, "r") as f:
        settings = yaml.safe_load(f) or {}
except (FileNotFoundError, yaml.YAMLError):
    settings = {}

img_size = 640  # default
try:
    img_size = settings['training']['yolo']['img_size']
except (KeyError, TypeError):
    pass

anomaly_threshold = (settings.get('anomaly', {})).get('threshold', 0.05)
autoencoder_path = BASE / "models" / "autoencoder.h5"

# Load the model once at module level, handling fallback
autoencoder_model = None
if autoencoder_path.exists():
    try:
        import tensorflow as tf
        autoencoder_model = tf.keras.models.load_model(str(autoencoder_path), compile=False)
    except ImportError:
        print("⚠️ TensorFlow is not installed. Anomaly detection will be disabled.")
    except Exception as e:
        print(f"Failed to load autoencoder model: {e}")

def detect_anomaly(image_bytes: bytes) -> dict:
    """
    Computes reconstruction error on the input image.
    If error > threshold, flags as Anomaly.
    """
    if autoencoder_model is None:
        return {
            "status": "PASS",
            "anomaly_score": 0.0,
            "threshold": anomaly_threshold,
            "message": "Autoencoder model not found. Passing by default."
        }

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess (Resize & Normalize)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    # Predict reconstruction
    reconstruction = autoencoder_model.predict(img_expanded, verbose=0)
    
    # Compute Mean Squared Error
    mse = np.mean(np.square(img_expanded - reconstruction))
    
    # Flag if error exceeds threshold
    is_anomaly = float(mse) > anomaly_threshold
    
    return {
        "status": "ANOMALY" if is_anomaly else "NORMAL",
        "anomaly_score": float(mse),
        "threshold": anomaly_threshold
    }
