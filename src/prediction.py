"""
pipeline: load model -> preprocess input -> predict -> return
class label + confidence.

This module is called by the FastAPI /predict endpoint.
It keeps a singleton model instance in memory to avoid reloading
on every request, which significantly reduces latency.
"""

from __future__ import annotations
import os, time, threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np

from .preprocessing import (
    preprocess_image_bytes,
    preprocess_image_path,
    CLASS_NAMES,
    CLASS_DISPLAY,
    NUM_CLASSES,
    IMG_SIZE,
)
from .model import PROD_MODEL_PATH, load_metadata

# Model singleton (lazy load, thread-safe)
_model = None
_model_lock = threading.Lock()
_model_loaded_at: Optional[str] = None
_model_path_in_use: Optional[str] = None

# Counters for API health/metrics
_prediction_count = 0
_last_prediction_at: Optional[str] = None
_counter_lock = threading.Lock()


def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_model(path: str = PROD_MODEL_PATH, force_reload: bool = False):
    """
    Load and return the Keras model.
    Uses lazy loading; reload if force_reload=True.
    """
    global _model, _model_loaded_at, _model_path_in_use
    if _model is not None and not force_reload:
        return _model

    with _model_lock:
        if _model is None or force_reload:
            import tensorflow as tf
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model not found at {path}. Train the model first."
                )
            _model = tf.keras.models.load_model(path)
            _model_loaded_at = _now_iso()
            _model_path_in_use = path
    return _model


def is_model_loaded() -> bool:
    """Return True if model is loaded in memory."""
    return _model is not None


def model_status() -> Dict:
    """Return status payload for /health endpoint."""
    meta = load_metadata()
    model_available = _model is not None or os.path.exists(PROD_MODEL_PATH)
    return {
        "model_loaded": _model is not None,
        "model_available": model_available,
        "model_path": _model_path_in_use or PROD_MODEL_PATH,
        "model_loaded_at": _model_loaded_at,
        "model_version": meta.get("version", "unknown"),
        "model_saved_at": meta.get("saved_at"),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "prediction_count": _prediction_count,
        "last_prediction_at": _last_prediction_at,
    }


def _record_prediction() -> None:
    """Update counters after a prediction."""
    global _prediction_count, _last_prediction_at
    with _counter_lock:
        _prediction_count += 1
        _last_prediction_at = _now_iso()


# Core prediction functions

def _format_result(probs: np.ndarray, elapsed_ms: float) -> Dict:
    """Convert softmax vector into API response payload."""
    probs = np.asarray(probs).ravel()
    idx = int(probs.argmax())
    confidence = float(probs[idx])

    ranked = sorted(
        (
            {
                "class_index": i,
                "class_name": CLASS_NAMES[i],
                "display_name": CLASS_DISPLAY[i],
                "probability": float(probs[i]),
            }
            for i in range(len(probs))
        ),
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
        "predicted_at": _now_iso(),
    }


def predict_from_bytes(data: bytes, model=None) -> Dict:
    """Predict from raw image bytes (API upload)."""
    model = model or get_model()
    x = preprocess_image_bytes(data)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_prediction()
    return _format_result(probs, elapsed_ms)


def predict_from_path(path: str, model=None) -> Dict:
    """Predict from image file path."""
    model = model or get_model()
    x = preprocess_image_path(path)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_prediction()
    return _format_result(probs, elapsed_ms)


def predict_batch(X: np.ndarray, model=None) -> List[Dict]:
    """Predict a batch of preprocessed images."""
    model = model or get_model()
    t0 = time.perf_counter()
    probs = model.predict(X, verbose=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_item = elapsed_ms / max(len(X), 1)
    results = [_format_result(p, per_item) for p in probs]
    with _counter_lock:
        globals()["_prediction_count"] += len(X)
        globals()["_last_prediction_at"] = _now_iso()
    return results


def reload_model(path: str = PROD_MODEL_PATH) -> Dict:
    """Reload model after retrain promotes a new version."""
    get_model(path=path, force_reload=True)
    return model_status()

# endpoint
if __name__ == "__main__":
    import json
    print("Prediction module self-check")
    print("-" * 50)
    print(json.dumps(model_status(), indent=2))
    fake = np.array([0.02, 0.91, 0.05, 0.02])
    print()
    print("Formatted result for synthetic softmax vector:")
    print(json.dumps(_format_result(fake, 12.34), indent=2)[:700])
