"""
cnn_lstm_service.py
===================
Inference service for the CNN + LSTM defect classifier.

Exposes:
  load_cnn_lstm_model()         → model (ready for inference)
  predict_cnn_lstm(model, img_bytes) → dict with verdict, confidence, heatmap
"""

import os
import io
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import base64

from cnn_lstm_model import build_model, CNNLSTM

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent   # project root
MODEL_PATH = BASE / "models" / "cnn_lstm_best.pth"
IMG_SIZE   = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Inference transform ────────────────────────────────────────────────────────
INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Model loader ──────────────────────────────────────────────────────────────
def load_cnn_lstm_model() -> CNNLSTM:
    """Load the trained CNN+LSTM model. Returns None if weights not found."""
    model = build_model(pretrained=True).to(DEVICE)

    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location=DEVICE)
        model.load_state_dict(state)
        print(f"✓ CNN+LSTM model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  CNN+LSTM weights not found at '{MODEL_PATH}'")
        print("   Run:  python cnn_lstm_train.py  to train the model first.")
        print("   Using untrained model for demo purposes.")

    model.eval()
    return model


# ── Global Grad-CAM cache (single instance to avoid hook leaks) ──
_gcam_instance = None

def _get_gradcam(model: CNNLSTM):
    """Return a cached GradCAM instance, re-creating only if model changed."""
    global _gcam_instance
    if _gcam_instance is None or _gcam_instance.model is not model:
        _gcam_instance = GradCAM(model)
    return _gcam_instance


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM on the last convolutional layer of the CNN backbone."""

    def __init__(self, model: CNNLSTM):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []  # track hooks for cleanup

        # Hook onto the last conv block in MobileNetV3 features
        target_layer = model.cnn[0][-1]  # last block in backbone.features
        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def cleanup(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns a Grad-CAM heatmap as a uint8 numpy array (H x W).

        img_tensor : [1, C, H, W]
        """
        self.model.zero_grad()
        img_tensor = img_tensor.to(DEVICE).requires_grad_(True)

        out = self.model(img_tensor)            # [1, 1]
        out[0, 0].backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Pool gradients over spatial dims
        grads  = self.gradients                          # [1, C, H, W]
        acts   = self.activations                        # [1, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam * 255).astype(np.uint8)
        return cam


def _apply_heatmap(orig_img: np.ndarray, cam: np.ndarray) -> str:
    """Overlay Grad-CAM on the original image. Returns base64 PNG string."""
    colormap  = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    colormap  = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    orig_rgb  = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    overlay   = (0.5 * orig_rgb + 0.5 * colormap).astype(np.uint8)
    pil       = Image.fromarray(overlay)
    buf       = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Main inference function ────────────────────────────────────────────────────
def predict_cnn_lstm(model: CNNLSTM, image_bytes: bytes) -> dict:
    """
    Run CNN+LSTM inference on raw image bytes.

    Returns
    -------
    dict:
      verdict          : 'GOOD' | 'DEFECTIVE'
      confidence       : float [0, 1]  — probability of being defective
      verdict_label    : 'PASS' | 'FAIL'
      heatmap_image    : base64 PNG string (Grad-CAM overlay)
      model_type       : 'CNN+LSTM'
      model_loaded     : bool
    """
    # Decode
    nparr    = np.frombuffer(image_bytes, np.uint8)
    img_cv   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("Could not decode the uploaded image.")
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Preprocess
    pil_img    = Image.fromarray(img_rgb)
    img_tensor = INFER_TF(pil_img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    # Grad-CAM + prediction (reuse cached instance to avoid hook leaks)
    gcam = _get_gradcam(model)
    with torch.enable_grad():
        cam = gcam.generate(img_tensor.clone())

    # Clean forward pass for final confidence
    with torch.no_grad():
        prob = model(img_tensor).item()  # P(defective)

    verdict       = "DEFECTIVE" if prob > 0.5 else "GOOD"
    verdict_label = "FAIL"      if prob > 0.5 else "PASS"

    # Build annotated preview (original + heatmap)
    img_resized   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    heatmap_b64   = _apply_heatmap(img_resized, cam)

    model_loaded  = MODEL_PATH.exists()

    return {
        "verdict":       verdict,
        "verdict_label": verdict_label,
        "confidence":    round(prob, 4),
        "heatmap_image": heatmap_b64,
        "model_type":    "CNN+LSTM",
        "model_loaded":  model_loaded,
        "analysis": {
            "defect_probability":  round(prob,       4),
            "healthy_probability": round(1.0 - prob, 4),
            "threshold":           0.5,
            "backbone":            "MobileNetV3-Small",
            "lstm_layers":         2,
            "lstm_hidden":         256,
        },
    }
