"""
Stage 1
prediction.py

Pipeline steps:
    1. Load the model
    2. Preprocess the input
    3. Make a prediction
    4. Return structured output (class label + confidence)

This module is used by the FastAPI /predict endpoint.
It keeps a single, shared instance of the model loaded at the module level.
"""

import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from preprocessing import (
    preprocess_image_bytes,
    preprocess_image_path,
    CLASS_NAMES,
    CLASS_DISPLAY,
    NUM_CLASSES,
    IMG_SIZE,
)
from .model import PROD_MODEL_PATH, load_metadata

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_model: Optional[tf.keras.Model] = None
_model_lock = threading.Lock()
_model_loaded_at: Optional[str] = None
_model_path_in_use: Optional[str] = None

# Counters for monitoring
_prediction_count = 0
_last_prediction_at: Optional[str] = None
_counter_lock = threading.Lock()


def _current_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_model(path: str = PROD_MODEL_PATH, force_reload: bool = False):
    """Load and return the Keras model (singleton)."""
    global _model, _model_loaded_at, _model_path_in_use

    if _model is not None and not force_reload:
        return _model

    with _model_lock:
        if _model is None or force_reload:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found at {path}. Train the model first.")
            _model = tf.keras.models.load_model(path)
            _model_loaded_at = _current_iso()
            _model_path_in_use = path
    return _model


def model_status() -> Dict:
    """Return model health/status info for monitoring endpoints."""
    meta = load_metadata()
    return {
        "loaded": _model is not None,
        "path": _model_path_in_use or PROD_MODEL_PATH,
        "loaded_at": _model_loaded_at,
        "version": meta.get("version", "unknown"),
        "saved_at": meta.get("saved_at"),
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "prediction_count": _prediction_count,
        "last_prediction_at": _last_prediction_at,
    }


def _record_prediction() -> None:
    """Update counters after each prediction."""
    global _prediction_count, _last_prediction_at
    with _counter_lock:
        _prediction_count += 1
        _last_prediction_at = _current_iso()


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def _format_result(probs: np.ndarray, elapsed_ms: float) -> Dict:
    """Convert raw probabilities into structured response."""
    probs = np.asarray(probs).ravel()
    idx = int(probs.argmax())
    confidence = float(probs[idx])

    ranked = sorted(
        [
            {
                "class_index": i,
                "class_name": CLASS_NAMES[i],
                "display_name": CLASS_DISPLAY[i],
                "probability": float(probs[i]),
            }
            for i in range(len(probs))
        ],
        key=lambda d: d["probability"],
        reverse=True,
    )

    return {
        "predicted_class_index": idx,
        "predicted_class": CLASS_NAMES[idx],
        "predicted_display_name": CLASS_DISPLAY[idx],
        "confidence": confidence,
        "is_healthy": CLASS_NAMES[idx] == "Healthy",
        "all_probabilities": ranked,
        "inference_time_ms": round(elapsed_ms, 2),
        "predicted_at": _current_iso(),
    }


def predict_from_bytes(data: bytes, model=None) -> Dict:
    """Predict from raw image bytes (used by API uploads)."""
    model = model or load_model()
    x = preprocess_image_bytes(data)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_prediction()
    return _format_result(probs, elapsed_ms)


def predict_from_path(path: str, model=None) -> Dict:
    """Predict from an image file path."""
    model = model or load_model()
    x = preprocess_image_path(path)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_prediction()
    return _format_result(probs, elapsed_ms)


def predict_batch(X: np.ndarray, model=None) -> List[Dict]:
    """Predict on a batch of preprocessed images."""
    model = model or load_model()
    t0 = time.perf_counter()
    probs = model.predict(X, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_item = elapsed_ms / max(len(X), 1)
    results = [_format_result(p, per_item) for p in probs]

    with _counter_lock:
        global _prediction_count, _last_prediction_at
        _prediction_count += len(X)
        _last_prediction_at = _current_iso()

    return results


def reload_model(path: str = PROD_MODEL_PATH) -> Dict:
    """Reload the model after retraining."""
    load_model(path=path, force_reload=True)
    return model_status()


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    print("Prediction module self-check")
    print(json.dumps(model_status(), indent=2))
    fake = np.array([0.02, 0.91, 0.05, 0.02])
    print(json.dumps(_format_result(fake, 12.34), indent=2)[:700])
