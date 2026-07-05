"""
app.py
======
Flask backend for the AI Defect Detection System.
Dual-model: YOLOv8 (object detection) + CNN+LSTM (binary classification).
Includes Transfer Learning, Anomaly Detection, and Active Learning integration.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from services.model_service import load_model, predict_defects, CLASS_INFO
from services.cnn_lstm_service import load_cnn_lstm_model, predict_cnn_lstm
from services.database import add_inspection as sql_add, get_history as sql_history, get_history_count as sql_count, get_history_stats as sql_stats, clear_history as sql_clear, export_history_csv
from services.firebase_db import add_inspection as fb_add, get_history as fb_history, get_history_count as fb_count, get_history_stats as fb_stats, clear_history as fb_clear, is_available as fb_available
from services.gemini_service import generate_chat_response
from anomaly.anomaly_service import detect_anomaly
from active_learning.active_learning_service import handle_detection, get_pending_images, label_image, PENDING_DIR

import os, json, uuid, cv2, numpy as np, webbrowser, threading, io, csv, subprocess, sys
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ── Load models at startup ─────────────────────────────────────
yolo_model     = load_model()
cnn_lstm_model = load_cnn_lstm_model()

BASE_DIR = Path(__file__).parent

# ── Database router: Firebase if available, else SQLite ─────
def _db_add(*a, **kw): return fb_add(*a, **kw) if fb_available() else sql_add(*a, **kw)
def _db_history(*a, **kw): return fb_history(*a, **kw) if fb_available() else sql_history(*a, **kw)
def _db_count(*a, **kw): return fb_count(*a, **kw) if fb_available() else sql_count(*a, **kw)
def _db_stats(*a, **kw): return fb_stats(*a, **kw) if fb_available() else sql_stats(*a, **kw)
def _db_clear(*a, **kw): return fb_clear(*a, **kw) if fb_available() else sql_clear(*a, **kw)

DATASETS = [
    {
        "name":        "DeepPCB",
        "description": "PCB defect dataset with bounding-box annotations",
        "classes":     ["crack", "blowhole", "break", "fray", "open", "short", "mousebite", "spur", "copper", "pin_hole"],
        "size":        "1 500 image pairs",
    }
]


def _get_defect_count(result):
    return len(result.get("defects", []))


def _get_verdict(result):
    return result.get("verdict_label") or result.get("verdict") or result.get("status", "UNKNOWN")


# ── Page Routes ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html", active="dashboard")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", active="analytics")

@app.route("/camera")
def camera():
    return render_template("camera.html", active="camera")

@app.route("/assistant")
def assistant():
    return render_template("assistant.html", active="assistant")

@app.route("/history")
def history_page():
    return render_template("history.html", active="history")

@app.route("/settings")
def settings():
    return render_template("settings.html", active="settings")


# ── Single Image Detection ───────────────────────────────────
@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        image_bytes = file.read()
        results = predict_defects(yolo_model, image_bytes)
        _db_add(file.filename, "YOLOv8", _get_verdict(results),
                       results.get("confidence", 0), _get_defect_count(results))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cnn-detect", methods=["POST"])
def cnn_detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        image_bytes = file.read()
        results = predict_cnn_lstm(cnn_lstm_model, image_bytes)
        _db_add(file.filename, "CNN+LSTM", _get_verdict(results),
                       results.get("confidence", 0), _get_defect_count(results))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Full Pipeline ────────────────────────────────────────────
@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        image_bytes = file.read()

        anomaly_result = detect_anomaly(image_bytes)
        yolo_results = predict_defects(yolo_model, image_bytes)

        al_status = handle_detection(image_bytes, yolo_results["confidence"], 0,
                                     has_defects=len(yolo_results.get("defects", [])) > 0)

        nparr  = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        verified_defects = []
        discarded_defects = []

        for defect in yolo_results.get("defects", []):
            x1, y1, x2, y2 = defect["bbox"]
            h, w = img_cv.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0: continue
            is_success, buffer = cv2.imencode(".jpg", crop)
            if not is_success: continue
            cnn_result = predict_cnn_lstm(cnn_lstm_model, buffer.tobytes())
            defect["verification"] = {
                "verdict": cnn_result["verdict"],
                "confidence": cnn_result["confidence"],
                "heatmap": cnn_result["heatmap_image"]
            }
            if cnn_result["verdict"] == "DEFECTIVE":
                verified_defects.append(defect)
            else:
                discarded_defects.append(defect)

        status = "FAIL" if len(verified_defects) > 0 or anomaly_result["status"] == "ANOMALY" else "PASS"

        final_results = {
            "status": status,
            "defects": verified_defects,
            "discarded_defects": discarded_defects,
            "confidence": yolo_results["confidence"],
            "annotated_image": yolo_results["annotated_image"],
            "anomaly_info": anomaly_result,
            "active_learning_flagged": al_status.get("status") == "FLAGGED",
            "model_info": {
                "type": "YOLOv8 + CNN-LSTM + TF-Anomaly",
                "yolo_classes": yolo_results["model_info"]["classes"]
            }
        }

        _db_add(file.filename, "Pipeline", _get_verdict(final_results),
                       final_results.get("confidence", 0), _get_defect_count(final_results), final_results)
        return jsonify(final_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Batch Upload ─────────────────────────────────────────────
@app.route("/api/batch-detect", methods=["POST"])
def batch_detect():
    try:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        results = []
        for f in files:
            if f.filename == "":
                continue
            img_bytes = f.read()
            yolo_res = predict_defects(yolo_model, img_bytes)
            anomaly_res = detect_anomaly(img_bytes)
            status = "FAIL" if len(yolo_res.get("defects", [])) > 0 or anomaly_res["status"] == "ANOMALY" else "PASS"
            r = {
                "filename": f.filename,
                "status": status,
                "defects": len(yolo_res.get("defects", [])),
                "confidence": yolo_res.get("confidence", 0),
            }
            _db_add(f.filename, "Batch", status, r["confidence"], r["defects"])
            results.append(r)

        passed = sum(1 for r in results if r["status"] == "PASS")
        return jsonify({
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Active Learning ──────────────────────────────────────────
@app.route("/api/pending-images")
def get_pending():
    return jsonify({"pending": get_pending_images()})

@app.route("/api/pending-images/<filename>")
def serve_pending_image(filename):
    return send_from_directory(PENDING_DIR, filename)

@app.route("/api/label", methods=["POST"])
def label_pending_image():
    try:
        data = request.json
        result = label_image(data.get("filename"), data.get("label_class", "0"), data.get("bboxes", []))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── History ──────────────────────────────────────────────────
@app.route("/api/history")
def history():
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    rows = _db_history(limit, offset)
    total = _db_count()
    return jsonify({"history": rows, "total": total})

@app.route("/api/history/stats")
def history_stats():
    return jsonify(_db_stats())

@app.route("/api/history/clear", methods=["POST"])
def history_clear():
    _db_clear()
    return jsonify({"status": "ok"})

@app.route("/api/history/export")
def history_export():
    csv_data = export_history_csv()
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspections.csv"})


# ── PDF Report ──────────────────────────────────────────────
@app.route("/api/report/pdf", methods=["POST"])
def report_pdf():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data"}), 400

        status = data.get("status") or data.get("verdict_label") or data.get("verdict", "UNKNOWN")
        defects = data.get("defects", [])
        conf = data.get("confidence", 0.0)
        model = (data.get("model_info") or {}).get("type") or data.get("model_type", "Unknown")
        is_pass = status in ("PASS", "GOOD")

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 40px; color: #1a1a2e; }}
  h1 {{ color: #0f3460; border-bottom: 3px solid #06B6D4; padding-bottom: 10px; }}
  .verdict {{ font-size: 28px; font-weight: 700; padding: 15px 25px; border-radius: 10px; text-align: center; }}
  .pass {{ background: #d1fae5; color: #065f46; }}
  .fail {{ background: #fee2e2; color: #991b1b; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; }}
  .footer {{ margin-top: 30px; font-size: 12px; color: #9ca3af; text-align: center; }}
</style></head><body>
<h1>🔬 DefectAI Pro — Inspection Report</h1>
<p><strong>Report ID:</strong> RPT-{uuid.uuid4().hex[:8].upper()}</p>
<p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><strong>Model:</strong> {model}</p>
<div class="verdict {'pass' if is_pass else 'fail'}">{'✅ PASS — Component Cleared' if is_pass else '❌ FAIL — Defects Detected'}</div>
<p><strong>Confidence:</strong> {round(float(conf) * 100, 1)}%</p>
<p><strong>Defects Found:</strong> {len(defects)}</p>"""

        if defects:
            html += """<table><thead><tr><th>#</th><th>Type</th><th>Confidence</th></tr></thead><tbody>"""
            for i, d in enumerate(defects, 1):
                html += f"<tr><td>{i}</td><td>{d['class']}</td><td>{round(d['confidence']*100,1)}%</td></tr>"
            html += "</tbody></table>"

        html += f"""<div class="footer">DefectAI Pro — AI-Powered PCB Inspection — Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

        try:
            import pdfkit
            pdf = pdfkit.from_string(html, False)
            return Response(pdf, mimetype="application/pdf",
                            headers={"Content-Disposition": f"attachment; filename=report_{uuid.uuid4().hex[:8]}.pdf"})
        except ImportError:
            from weasyprint import HTML
            pdf = HTML(string=html).write_pdf()
            return Response(pdf, mimetype="application/pdf",
                            headers={"Content-Disposition": f"attachment; filename=report_{uuid.uuid4().hex[:8]}.pdf"})
        except Exception:
            return jsonify({"report_html": html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Classes ─────────────────────────────────────────────────
@app.route("/api/classes")
def classes():
    return jsonify({str(k): v for k, v in CLASS_INFO.items()})


# ── Gemini AI Assistant Chat ────────────────────────────────
@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Gather context for Gemini
        context = {
            "stats": _db_stats(),
            "health": {
                "yolo_model": "custom trained" if os.path.exists("models/best.pt") else "fallback",
                "cnn_lstm_model": "trained" if os.path.exists("models/cnn_lstm_best.pth") else "untrained",
                "anomaly_model": "loaded" if os.path.exists("models/autoencoder.h5") else "not loaded",
            },
            "recent_history": _db_history(limit=10),
        }

        response = generate_chat_response(user_message, context)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"❌ Error: {str(e)}"}), 500


# ── Favicon ─────────────────────────────────────────────────
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"),
                               "favicon.svg", mimetype="image/svg+xml")


# ── IP Camera Proxy ─────────────────────────────────────────
@app.route("/api/camera-proxy", methods=["POST"])
def camera_proxy():
    data = request.json or {}
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"error": "Could not read frame"}), 502
        _, buffer = cv2.imencode(".jpg", frame)
        return Response(buffer.tobytes(), mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health ──────────────────────────────────────────────────
@app.route("/health")
def health():
    yolo_status = "custom trained" if os.path.exists("models/best.pt") else "fallback"
    cnn_status = "trained" if os.path.exists("models/cnn_lstm_best.pth") else "untrained"
    stats = _db_stats()
    return jsonify({
        "status": "ok",
        "yolo_model": yolo_status,
        "cnn_lstm_model": cnn_status,
        "anomaly_model": "loaded" if os.path.exists("models/autoencoder.h5") else "not loaded",
        "total_inspections": stats["total"],
        "pass_rate": round(stats["passed"] / stats["total"] * 100, 1) if stats["total"] else 0,
    })


# ── Report (JSON) ──────────────────────────────────────────
@app.route("/api/report", methods=["POST"])
def generate_report():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        status = data.get("status") or data.get("verdict_label") or data.get("verdict", "UNKNOWN")
        defects = data.get("defects", [])
        conf = data.get("confidence", 0.0)
        model = (data.get("model_info") or {}).get("type") or data.get("model_type", "Unknown")
        is_pass = status in ("PASS", "GOOD")
        report = {
            "report_id": f"RPT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(),
            "model_used": model,
            "verdict": "PASS" if is_pass else "FAIL",
            "confidence": round(float(conf), 4),
            "num_defects": len(defects),
            "defect_classes": list({d["class"] for d in defects}),
            "anomaly_status": (data.get("anomaly_info") or {}).get("status", "N/A"),
            "active_learning_flagged": data.get("active_learning_flagged", False),
            "recommendation": (
                "Component passed inspection. Clear for assembly." if is_pass
                else f"{len(defects)} defect(s) detected. Remove component from production line."
            ),
        }
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Training Trigger ────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def trigger_training():
    try:
        data = request.json or {}
        model_type = data.get("model", "yolo")

        if model_type == "yolo":
            script = str(BASE_DIR / "training" / "yolo_train.py")
        elif model_type == "cnn_lstm":
            script = str(BASE_DIR / "training" / "cnn_lstm_train.py")
        elif model_type == "autoencoder":
            script = str(BASE_DIR / "anomaly" / "anomaly_model.py")
        else:
            return jsonify({"error": "Unknown model type"}), 400

        if not os.path.exists(script):
            return jsonify({"error": f"Training script not found: {script}"}), 404

        def run_training():
            try:
                result = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=str(BASE_DIR))
                log_path = BASE_DIR / "training" / "training_log.txt"
                with open(log_path, "w") as f:
                    f.write(f"=== Training {model_type} at {datetime.now()} ===\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write(f"\n--- stderr ---\n{result.stderr}")
                print(f"Training {model_type} completed. Log saved to {log_path}")
            except Exception as e:
                print(f"Training failed: {e}")

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return jsonify({"status": "started", "model": model_type, "message": f"Training {model_type} started in background"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/status")
def training_status():
    log_path = BASE_DIR / "training" / "training_log.txt"
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        last_line = lines[-1].strip() if lines else ""
        return jsonify({"status": "completed", "last_log": last_line})
    return jsonify({"status": "idle"})


# ── Start ──────────────────────────────────────────────────
def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    print("=" * 55)
    print("  DefectAI Pro -- Starting server")
    print("  Dashboard: http://localhost:5000")
    print("  Camera:    http://localhost:5000/camera")
    print("  Analytics: http://localhost:5000/analytics")
    print("  Assistant: http://localhost:5000/assistant")
    print("  History:   http://localhost:5000/history")
    print("  Settings:  http://localhost:5000/settings")
    print("  Batch API: POST /api/batch-detect")
    print("  CSV Export: GET /api/history/export")
    print("=" * 55)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)