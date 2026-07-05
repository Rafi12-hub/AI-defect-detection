"""
firebase_db.py
==============
Firebase Firestore integration via REST API.
Uses the web API key + project ID (no service account needed).
Falls back to SQLite if Firebase is unavailable.
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent

# ── Firebase Web Config ────────────────────────────────────
PROJECT_ID = "gitam-3395d"
API_KEY = "AIzaSyADysCuhZGVhUAmO_ReJQE2MoIpIe8RniM"
FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"

_available = None


FIREBASE_ENABLE_URL = "https://console.developers.google.com/apis/library/firestore.googleapis.com?project=gitam-3395d"

def is_available():
    global _available
    if _available is not None:
        return _available
    try:
        r = requests.get(
            f"{FIRESTORE_URL}/inspections?pageSize=1&key={API_KEY}",
            timeout=5,
        )
        if r.status_code == 403:
            print(f"[firebase_db] ⚠️ Firestore API not enabled. Enable it at: {FIREBASE_ENABLE_URL}")
            _available = False
            return False
        _available = r.status_code < 500
        return _available
    except Exception:
        _available = False
        return False


def _doc_to_dict(doc):
    """Convert a Firestore document to a flat dict."""
    fields = doc.get("fields", {})
    result = {"id": doc.get("name", "").split("/")[-1]}
    for key, val in fields.items():
        if "stringValue" in val:
            result[key] = val["stringValue"]
        elif "integerValue" in val:
            result[key] = int(val["integerValue"])
        elif "doubleValue" in val:
            result[key] = float(val["doubleValue"])
        elif "booleanValue" in val:
            result[key] = val["booleanValue"]
        elif "timestampValue" in val:
            result[key] = val["timestampValue"]
        else:
            result[key] = str(val)
    return result


def _dict_to_fields(data):
    """Convert a flat dict to Firestore fields format."""
    fields = {}
    for key, val in data.items():
        if val is None:
            continue
        if isinstance(val, bool):
            fields[key] = {"booleanValue": val}
        elif isinstance(val, int):
            fields[key] = {"integerValue": str(val)}
        elif isinstance(val, float):
            fields[key] = {"doubleValue": val}
        elif isinstance(val, str):
            fields[key] = {"stringValue": val}
        else:
            fields[key] = {"stringValue": str(val)}
    return fields


def _req(method, path, data=None):
    """Make a Firestore REST API request."""
    separator = "&" if "?" in path else "?"
    url = f"{FIRESTORE_URL}/{path}{separator}key={API_KEY}"
    try:
        r = requests.request(method, url, json=data, timeout=10)
        if r.status_code >= 500:
            print(f"[firebase_db] Server error {r.status_code}: {r.text[:200]}")
            return None
        if r.status_code >= 400:
            print(f"[firebase_db] Error {r.status_code}: {r.text[:200]}")
            return None
        return r.json() if r.text else None
    except requests.exceptions.ConnectionError:
        print("[firebase_db] Cannot reach Firebase (offline?)")
        return None
    except Exception as e:
        print(f"[firebase_db] Request error: {e}")
        return None


# ── CRUD Operations ────────────────────────────────────────

def add_inspection(filename, model, verdict, confidence, num_defects, details=None):
    """Add an inspection record to Firestore."""
    data = {
        "fields": _dict_to_fields({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "model": model,
            "verdict": verdict,
            "confidence": float(confidence),
            "num_defects": int(num_defects),
        })
    }
    if details:
        data["fields"]["details"] = {"stringValue": json.dumps(details) if isinstance(details, dict) else str(details)}

    # Use POST to auto-generate document ID, or PUT with specific ID
    result = _req("POST", "inspections", data)
    if result and "name" in result:
        return result["name"].split("/")[-1]
    return None


def get_history(limit=100, offset=0):
    """Fetch inspection history from Firestore."""
    result = _req("GET", f"inspections?pageSize={limit}")
    if not result or "documents" not in result:
        return []
    docs = result.get("documents", [])
    items = []
    for doc in docs:
        item = _doc_to_dict(doc)
        items.append({
            "id": item.get("id", ""),
            "timestamp": item.get("timestamp", ""),
            "filename": item.get("filename", ""),
            "model": item.get("model", ""),
            "verdict": item.get("verdict", ""),
            "confidence": item.get("confidence", 0),
            "num_defects": item.get("num_defects", 0),
        })
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return items[offset:offset + limit]


def get_history_count():
    result = _req("GET", "inspections?pageSize=1")
    if result:
        # Estimate count from existence
        docs = result.get("documents", [])
        return len(docs)
    return 0


def get_history_stats():
    result = _req("GET", "inspections?pageSize=1000")
    if not result or "documents" not in result:
        return {"total": 0, "passed": 0, "failed": 0, "avg_confidence": 0}
    docs = result.get("documents", [])
    total = 0
    passed = 0
    conf_sum = 0
    for doc in docs:
        item = _doc_to_dict(doc)
        total += 1
        if item.get("verdict") not in ("FAIL", "DEFECTIVE", "ANOMALY"):
            passed += 1
        conf_sum += float(item.get("confidence", 0))
    failed = total - passed
    avg_conf = conf_sum / total if total else 0
    return {"total": total, "passed": passed, "failed": failed, "avg_confidence": round(avg_conf, 4)}


def clear_history():
    """Delete all inspection documents from Firestore."""
    result = _req("GET", "inspections?pageSize=1000")
    if not result or "documents" not in result:
        return
    for doc in result.get("documents", []):
        doc_id = doc.get("name", "").split("/")[-1]
        _req("DELETE", f"inspections/{doc_id}")
