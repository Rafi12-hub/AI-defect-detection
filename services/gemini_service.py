"""
gemini_service.py
==================
Google Gemini API integration for the AI Assistant.
Uses the inspection context to generate intelligent responses.
"""

import os
import yaml
from google import genai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load API Key ────────────────────────────────────────────
def _load_api_key():
    key_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
    if os.path.exists(key_path):
        with open(key_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("gemini", {}).get("api_key", "")
    return os.environ.get("GEMINI_API_KEY", "")

API_KEY = _load_api_key()

# ── Configure Gemini ────────────────────────────────────────
_client = None
if API_KEY:
    _client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash"

# ── System Prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """You are DefectAI Assistant, an expert AI assistant for a PCB (Printed Circuit Board) defect detection system.

Your capabilities:
- You have REAL-TIME access to inspection data (history, stats, system health)
- Answer questions about defect analysis, pass rates, model confidence
- Provide insights and recommendations based on inspection trends
- Keep answers CONCISE and TECHNICAL (2-4 sentences ideally)
- Use emojis sparingly but appropriately

When asked about specific data, refer to the context provided.
If you don't know something, say so clearly — do NOT make up data.

Context from the system will be provided with each request."""


def generate_chat_response(user_message: str, context: dict = None) -> str:
    """
    Send a message to Gemini and get a response.
    context: dict with keys like 'history', 'stats', 'health', etc.
    """
    if not API_KEY or not _client:
        return (
            "⚠️ Gemini API key not configured.\n\n"
            "Set it in `config/api_keys.yaml`:\n"
            "```yaml\n"
            "gemini:\n"
            "  api_key: \"YOUR_KEY\"\n"
            "```\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    try:
        # Build context string
        context_str = ""
        if context:
            parts = []
            if context.get("stats"):
                s = context["stats"]
                parts.append(
                    f"System Stats — Total inspections: {s.get('total', 0)}, "
                    f"Passed: {s.get('passed', 0)}, Failed: {s.get('failed', 0)}, "
                    f"Avg confidence: {s.get('avg_confidence', 0) * 100:.1f}%"
                )
            if context.get("health"):
                h = context["health"]
                parts.append(
                    f"System Health — YOLO: {h.get('yolo_model', '?')}, "
                    f"CNN+LSTM: {h.get('cnn_lstm_model', '?')}, "
                    f"Anomaly: {h.get('anomaly_model', '?')}"
                )
            if context.get("recent_history"):
                rows = context["recent_history"][:5]
                for r in rows:
                    parts.append(
                        f"Inspection — File: {r.get('filename', '?')}, "
                        f"Verdict: {r.get('verdict', '?')}, "
                        f"Confidence: {r.get('confidence', 0) * 100:.1f}%, "
                        f"Defects: {r.get('num_defects', 0)}"
                    )
            if parts:
                context_str = "Here is the current system context:\n" + "\n".join(parts)

        full_prompt = f"{context_str}\n\nUser question: {user_message}" if context_str else user_message

        chat = _client.chats.create(
            model=MODEL_NAME,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        response = chat.send_message(full_prompt)
        return response.text

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return (
                "⚠️ Gemini API quota exhausted. The free tier has daily limits.\n\n"
                "To fix: Create a new key at https://aistudio.google.com/apikey\n"
                "Or wait a few minutes and try again."
            )
        if "API_KEY_INVALID" in err or "API key" in err:
            return "⚠️ Invalid Gemini API key. Check `config/api_keys.yaml`."
        return f"❌ Gemini error: {err[:200]}"
