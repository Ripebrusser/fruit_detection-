"""Flask backend for the AI-Based Fruit Detection and Freshness Analysis website.

The app is a thin, defensive layer around :class:`SRC.models.predictor.FruitPredictor`.
It never invents results: every number returned by ``/api/analyze`` comes from the
trained model, and unavailable features (shelf life, validated cross-section support)
are reported as unavailable.

Run from the project root:
    python -m Website.app
    python -m Website.app --port 5000
"""
from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python Website/app.py` as well as -m
    sys.path.insert(0, str(PROJECT_ROOT))

WEBSITE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WEBSITE_DIR / "static" / "uploads"
METRICS_PATH = PROJECT_ROOT / "Evaluation" / "metrics" / "metrics.json"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_PIXELS = 40_000_000            # ~40 MP guard against decompression bombs
MIN_SIDE = 64
UPLOAD_TTL_SECONDS = 60 * 60

_predictor = None
_predictor_error: str | None = None
_predictor_lock = threading.Lock()


def get_predictor():
    """Load the model once, on first use, and remember a failure instead of retrying."""
    global _predictor, _predictor_error
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        if _predictor is None and _predictor_error is None:
            try:
                import numpy as np

                from SRC.models.predictor import FruitPredictor

                loaded = FruitPredictor()
                # Warm the graph once so the first real user does not wait for lazy init.
                loaded.predict_array(np.full((224, 224, 3), 128, dtype=np.uint8))
                _predictor = loaded
            except Exception as exc:  # missing model, broken environment, ...
                _predictor_error = f"{type(exc).__name__}: {exc}"
    return _predictor


def cleanup_uploads(max_age_seconds: int = UPLOAD_TTL_SECONDS) -> None:
    """Delete stale uploads so user photographs do not accumulate on disk."""
    if not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - max_age_seconds
    for path in UPLOAD_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        UPLOAD_DIR=str(UPLOAD_DIR),
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024))

    # ------------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        predictor = get_predictor()
        return jsonify(
            {
                "ok": predictor is not None,
                "model_loaded": predictor is not None,
                "error": _predictor_error,
            }
        ), (200 if predictor is not None else 503)

    @app.get("/api/model-info")
    def model_info():
        """Transparent model card numbers, straight from the evaluation run."""
        if not METRICS_PATH.exists():
            return jsonify({"ok": False, "error": "evaluation metrics not available yet"}), 503
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        predictor = get_predictor()
        meta = getattr(predictor, "class_names", None)
        return jsonify(
            {
                "ok": True,
                "model": metrics.get("model"),
                "test_accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("f1_macro"),
                "condition_level_accuracy": metrics.get("condition_level_accuracy"),
                "fruit_level_accuracy": metrics.get("fruit_level_accuracy"),
                "test_split": metrics.get("test_split"),
                "calibration": metrics.get("calibration"),
                "classes": meta,
                "supported_fruits": getattr(predictor, "fruits", None),
                "shelf_life_supported": False,
                "cross_section_validated": False,
            }
        )

    @app.post("/api/analyze")
    def analyze():
        predictor = get_predictor()
        if predictor is None:
            return jsonify(
                {
                    "ok": False,
                    "error": "The prediction model is not available on this server.",
                    "detail": _predictor_error,
                }
            ), 503

        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"ok": False, "error": "No image was uploaded."}), 400

        safe_name = secure_filename(uploaded.filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify(
                {
                    "ok": False,
                    "error": "Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image.",
                }
            ), 400

        payload = uploaded.read()
        if not payload:
            return jsonify({"ok": False, "error": "The uploaded file is empty."}), 400

        try:
            with Image.open(io.BytesIO(payload)) as probe:
                probe.verify()  # structural check; does not decode pixels
            image = Image.open(io.BytesIO(payload))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"ok": False, "error": "That file is not a readable image."}), 400

        width, height = image.size
        if width * height > MAX_PIXELS:
            return jsonify(
                {"ok": False, "error": f"Image is too large ({width}x{height}). Maximum 40 megapixels."}
            ), 400
        if min(width, height) < MIN_SIDE:
            return jsonify(
                {"ok": False, "error": f"Image is too small ({width}x{height}). Minimum side is {MIN_SIDE}px."}
            ), 400

        image = image.convert("RGB")

        try:
            result = predictor.analyze(image, source_name=safe_name)
        except Exception as exc:  # model/runtime failure must be visible, not silent
            app.logger.exception("inference failed")
            return jsonify({"ok": False, "error": "Inference failed.", "detail": str(exc)}), 500

        cleanup_uploads()
        stored_name = f"{int(time.time() * 1000)}_{safe_name.rsplit('.', 1)[0][:40]}{extension}"
        stored_name = secure_filename(stored_name) or f"upload{extension}"
        destination = Path(app.config["UPLOAD_DIR"]) / stored_name
        destination.write_bytes(payload)

        result["image_url"] = url_for("static", filename=f"uploads/{stored_name}")
        result["image_name"] = safe_name
        return jsonify(result)

    @app.delete("/api/uploads/<path:name>")
    def delete_upload(name: str):
        """Clear/reset: remove one stored upload (only inside the upload folder)."""
        candidate = Path(app.config["UPLOAD_DIR"]) / secure_filename(name)
        upload_dir = Path(app.config["UPLOAD_DIR"]).resolve()
        if candidate.resolve().parent != upload_dir:
            return jsonify({"ok": False, "error": "Invalid file reference."}), 400
        if candidate.exists():
            candidate.unlink()
            return jsonify({"ok": True, "deleted": candidate.name})
        return jsonify({"ok": False, "error": "File not found."}), 404

    # ---------------------------------------------------------------- errors
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(
            {"ok": False, "error": f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}
        ), 413

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        """Keep real HTTP status codes (405, 400, ...) instead of turning them into 500s."""
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": exc.description}), exc.code
        return exc

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Unknown endpoint."}), 404
        return render_template("index.html", max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024)), 404

    @app.errorhandler(Exception)
    def unhandled(exc):  # pragma: no cover - defensive
        app.logger.exception("unhandled error")
        return jsonify({"ok": False, "error": "Unexpected server error."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the fruit detection website")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
